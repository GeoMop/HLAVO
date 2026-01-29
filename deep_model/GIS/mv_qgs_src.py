#!/usr/bin/env python3
"""
mv_qgis_src.py src_dir dst_dir [--force] [--dry-run]

- Expects exactly one *.qgs in current directory
- Scans .qgs for resource paths under src_dir
- Copies resources into dst_dir (preserving relative structure)
- Shapefile packs: if ANY sidecar from a shapefile pack is referenced, convert the pack to .gpkg
- GeoTIFF: copy; if .tfw exists, try to embed transform into copied tif (else keep .tfw as sidecar)
- Other files: copy with sidecars (same folder: <stem>.* OR <filename>.*)
- Rewrites paths in the .qgs and writes a .bak backup
- Quiet on success; reports failures only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple


_DELIMS = r"""[|<"'\n\r&]"""

# Recognize "files that belong to a shapefile pack"
# includes "foo.shp.xml" metadata
SHP_PACK_EXTS = {
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj",
    ".sbn", ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".ixs", ".mxs", ".atx",
    ".shp.xml",
}

# World file candidates for GeoTIFF
WORLD_EXTS = [".tfw", ".tifw", ".tiffw", ".wld"]


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def is_shp_pack_member(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith(".shp.xml"):
        return True
    return p.suffix.lower() in SHP_PACK_EXTS


def single_qgs(cwd: Path) -> Path:
    qgs = sorted(cwd.glob("*.qgs"))
    if not qgs:
        die("No .qgs found in current directory.")
    if len(qgs) > 1:
        die("More than one .qgs found: " + ", ".join(p.name for p in qgs))
    return qgs[0]


def abs_from_cwd(cwd: Path, p: str | Path) -> Path:
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = cwd / pp
    return pp.resolve(strict=False)


def build_src_variants(cwd: Path, src_arg: str, src_abs: Path) -> List[str]:
    typed = src_arg.rstrip("/")
    norm = os.path.normpath(src_arg).rstrip("/")
    try:
        rel = str(src_abs.relative_to(cwd)).rstrip("/")
    except ValueError:
        rel = ""
    abs_s = str(src_abs).rstrip("/")

    out: List[str] = []
    for v in (typed, norm, rel, abs_s):
        if v and v not in out:
            out.append(v)
    return out


def choose_dst_variant(old_src: str, dst_arg: str, dst_abs: Path, cwd: Path) -> str:
    dst_typed = dst_arg.rstrip("/")
    try:
        dst_rel = str(dst_abs.relative_to(cwd)).rstrip("/")
    except ValueError:
        dst_rel = ""
    dst_abs_s = str(dst_abs).rstrip("/")
    if old_src.startswith("/"):
        return dst_abs_s
    return dst_rel or dst_typed


def iter_qgs_resources(qgs_text: str, srcs: List[str]) -> Iterator[str]:
    seen: Set[str] = set()
    for src in srcs:
        pat = re.compile(re.escape(src) + r"(?:/[^|<\"'\n\r&]*)?")
        for m in pat.finditer(qgs_text):
            s = m.group(0).strip()
            if not s or s == src or s in seen:
                continue
            seen.add(s)
            yield s


def raw_to_abs_path(cwd: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = cwd / p
    return p.resolve(strict=False)


def iter_sidecars(primary_abs: Path) -> Iterator[Path]:
    parent = primary_abs.parent
    stem_prefix = primary_abs.stem + "."
    name_prefix = primary_abs.name + "."

    yield primary_abs

    try:
        for ch in parent.iterdir():
            if not ch.is_file():
                continue
            n = ch.name
            if n == primary_abs.name:
                continue
            if n.startswith(stem_prefix) or n.startswith(name_prefix):
                yield ch.resolve(strict=False)
    except FileNotFoundError:
        return


def dst_for_src(src_file: Path, src_root: Path, dst_root: Path) -> Optional[Path]:
    try:
        rel = src_file.relative_to(src_root)
    except ValueError:
        return None
    return dst_root / rel


def copy_file_quiet(src: Path, dst: Path, force: bool, dry_run: bool) -> Optional[str]:
    """
    Returns failure message if failed, else None.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force:
            return f"dst exists (use --force): {dst}"
        if dry_run:
            return None
        if dst.exists() and force:
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.copy2(src, dst)
        return None
    except Exception as e:
        return f"copy failed: {src} -> {dst}: {e}"


# ---------- Shapefile pack handling ----------

def shp_from_pack_member(p: Path) -> Path:
    """
    Given any pack member, return corresponding .shp path by basename.
    Handles foo.shp.xml too.
    """
    if p.name.lower().endswith(".shp.xml"):
        # strip the trailing ".xml" only
        return p.with_name(p.name[:-4])  # remove ".xml" => foo.shp
    return p.with_suffix(".shp")


def build_dst_path_like(raw: str, dst_root: Path, src_root: Path, member_abs: Path, cwd: Path) -> Tuple[str, str]:
    """
    Returns (old_style_path_for_dst_member, style_kind) where style_kind is 'abs' or 'rel'.
    Used to create new gpkg path string matching the style used in the project.
    """
    # If raw path was absolute in project, output absolute.
    if raw.startswith("/"):
        return str(dst_root.resolve(strict=False)), "abs"
    # Else keep relative if possible
    try:
        return str(dst_root.resolve(strict=False).relative_to(cwd)), "rel"
    except ValueError:
        # fallback: use as-typed dst_root (relative) behavior
        return str(dst_root), "rel"


def shp_to_gpkg(
    src_shp: Path,
    dst_gpkg: Path,
    layername: str,
    force: bool,
    dry_run: bool,
) -> Optional[str]:
    """
    Convert SHP -> GPKG. Returns failure message or None on success.
    """
    try:
        dst_gpkg.parent.mkdir(parents=True, exist_ok=True)
        if dst_gpkg.exists() and not force:
            return f"dst exists (use --force): {dst_gpkg}"
        if dry_run:
            return None
        if dst_gpkg.exists() and force:
            dst_gpkg.unlink()

        # Prefer python GDAL
        try:
            from osgeo import gdal  # type: ignore

            # VectorTranslate can return None on failure without raising, so check result
            res = gdal.VectorTranslate(
                destNameOrDestDS=str(dst_gpkg),
                srcDS=str(src_shp),
                format="GPKG",
                layerName=layername,
            )
            if res is None:
                return "GDAL VectorTranslate returned None"
            return None
        except Exception as e:
            # Fall back to ogr2ogr
            cmd = ["ogr2ogr", "-f", "GPKG", str(dst_gpkg), str(src_shp), "-nln", layername]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                return f"ogr2ogr failed: {res.stderr.strip()}"
            return None

    except FileNotFoundError as e:
        return f"missing tool or file: {e}"
    except Exception as e:
        return f"conversion error: {e}"


# ---------- GeoTIFF + TFW embedding ----------

def find_world_file(src_tif: Path) -> Optional[Path]:
    parent = src_tif.parent
    stem = src_tif.stem
    # typical: <stem>.tfw etc
    for ext in WORLD_EXTS:
        p = parent / f"{stem}{ext}"
        if p.is_file():
            return p.resolve(strict=False)
    # also allow <filename>.<ext> (rare)
    for ext in WORLD_EXTS:
        p = parent / f"{src_tif.name}{ext}"
        if p.is_file():
            return p.resolve(strict=False)
    # uppercase variants on case-sensitive fs
    for ext in ["TFW", "TIFW", "TIFFW", "WLD"]:
        p = parent / f"{stem}.{ext}"
        if p.is_file():
            return p.resolve(strict=False)
    return None


def worldfile_to_geotransform(wld_path: Path) -> Tuple[float, float, float, float, float, float]:
    lines = wld_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(lines) < 6:
        raise ValueError("world file has <6 lines")
    A = float(lines[0].strip())
    D = float(lines[1].strip())
    B = float(lines[2].strip())
    E = float(lines[3].strip())
    C = float(lines[4].strip())
    F = float(lines[5].strip())
    GT0 = C - (A / 2.0) - (B / 2.0)
    GT3 = F - (D / 2.0) - (E / 2.0)
    return (GT0, A, B, GT3, D, E)


def embed_worldfile_into_tif(dst_tif: Path, wld_path: Path, dry_run: bool) -> Optional[str]:
    """
    Returns failure message or None on success.
    """
    if dry_run:
        return None
    try:
        from osgeo import gdal  # type: ignore

        gt = worldfile_to_geotransform(wld_path)
        ds = gdal.Open(str(dst_tif), gdal.GA_Update)
        if ds is None:
            return "GDAL could not open GeoTIFF for update (locked? permissions?)"
        ds.SetGeoTransform(gt)
        ds.FlushCache()
        ds = None
        return None
    except Exception as e:
        return str(e)


# ---------- Project rewrite ----------

def rewrite_project(
    qgs_path: Path,
    original_text: str,
    srcs: List[str],
    dst_arg: str,
    dst_abs: Path,
    cwd: Path,
    rewrites: Dict[str, str],
    dry_run: bool,
    failures: List[str],
) -> None:
    backup = qgs_path.with_suffix(qgs_path.suffix + ".bak")

    # base src->dst replacements
    pairs: List[Tuple[str, str]] = [(s, choose_dst_variant(s, dst_arg, dst_abs, cwd)) for s in srcs]
    pairs.sort(key=lambda t: len(t[0]), reverse=True)

    text = original_text
    for old, new in pairs:
        text = text.replace(old, new)

    # datasource-specific rewrites (e.g. .shp -> .gpkg|layername=...)
    for old, new in sorted(rewrites.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(old, new)

    if text == original_text:
        return

    if dry_run:
        return

    try:
        shutil.copy2(str(qgs_path), str(backup))
        qgs_path.write_text(text, encoding="utf-8")
    except Exception as e:
        failures.append(f"project rewrite failed: {qgs_path}: {e}")


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cwd = Path.cwd()
    qgs = single_qgs(cwd)
    qgs_text = qgs.read_text(encoding="utf-8", errors="replace")

    src_root = abs_from_cwd(cwd, args.src_dir)
    dst_root = abs_from_cwd(cwd, args.dst_dir)

    if not src_root.is_dir():
        die(f"src_dir is not a directory: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    srcs = build_src_variants(cwd, args.src_dir, src_root)

    failures: List[str] = []
    rewrites: Dict[str, str] = {}

    copied: Set[Path] = set()
    converted_shp_stems: Set[Path] = set()  # key by source .shp path (absolute)

    for raw in iter_qgs_resources(qgs_text, srcs):
        abs_p = raw_to_abs_path(cwd, raw)

        # Missing reference: report and continue
        if not abs_p.is_file():
            # Special case: if it's a shapefile pack sidecar, try to resolve .shp and proceed
            if is_shp_pack_member(abs_p):
                shp = shp_from_pack_member(abs_p)
                if not shp.is_file():
                    failures.append(f"missing shapefile pack (referenced {raw}, expected {shp})")
                    continue
                abs_p = shp
            else:
                failures.append(f"missing file referenced in project: {raw}")
                continue

        # Shapefile pack conversion triggered by ANY pack member
        if is_shp_pack_member(abs_p):
            shp = shp_from_pack_member(abs_p)
            if not shp.is_file():
                failures.append(f"missing .shp for pack member {abs_p}")
                continue

            if shp in converted_shp_stems:
                # already converted; still rewrite this raw path if it appears in project
                pass
            else:
                dst_like = dst_for_src(shp, src_root, dst_root)
                if dst_like is None:
                    continue
                dst_gpkg = dst_like.with_suffix(".gpkg")
                layername = shp.stem

                err = shp_to_gpkg(shp, dst_gpkg, layername=layername, force=args.force, dry_run=args.dry_run)
                if err:
                    failures.append(f"SHP->GPKG failed for {shp}: {err}")
                    # If conversion fails, fall back to copying the pack (quiet), but report conversion failure.
                    for f in iter_sidecars(shp):
                        if not f.is_file() or f in copied:
                            continue
                        dst_f = dst_for_src(f, src_root, dst_root)
                        if dst_f is None:
                            continue
                        msg = copy_file_quiet(f, dst_f, force=args.force, dry_run=args.dry_run)
                        if msg:
                            failures.append(msg)
                        copied.add(f)
                else:
                    converted_shp_stems.add(shp)

            # Rewrite: any referenced pack member path should point to gpkg layer
            dst_like = dst_for_src(shp, src_root, dst_root)
            if dst_like is not None:
                dst_gpkg = dst_like.with_suffix(".gpkg")
                # keep same path style (abs vs rel) as the original raw
                if raw.startswith("/"):
                    gpkg_path_str = str(dst_gpkg.resolve(strict=False))
                else:
                    try:
                        gpkg_path_str = str(dst_gpkg.resolve(strict=False).relative_to(cwd))
                    except ValueError:
                        # fallback to typed dst_dir + rel
                        rel = shp.relative_to(src_root)
                        gpkg_path_str = str(Path(args.dst_dir) / rel).replace(".shp", ".gpkg")
                new_uri = f"{gpkg_path_str}|layername={shp.stem}"

                # Replace both the original raw and the post-base-replacement variant
                post_base = raw
                for s in srcs:
                    if s and post_base.startswith(s):
                        post_base = post_base.replace(s, choose_dst_variant(s, args.dst_dir, dst_root, cwd), 1)
                        break
                rewrites[raw] = new_uri
                rewrites[post_base] = new_uri

            continue  # shapefile handled

        # GeoTIFF special: copy tif, then attempt embed tfw
        if abs_p.suffix.lower() in (".tif", ".tiff"):
            dst_tif = dst_for_src(abs_p, src_root, dst_root)
            if dst_tif is None:
                continue

            if abs_p not in copied:
                msg = copy_file_quiet(abs_p, dst_tif, force=args.force, dry_run=args.dry_run)
                if msg:
                    failures.append(msg)
                copied.add(abs_p)

            wld = find_world_file(abs_p)
            embedded_ok = False
            if wld is not None and wld.is_file():
                err = embed_worldfile_into_tif(dst_tif, wld, dry_run=args.dry_run)
                if err:
                    failures.append(f"TFW embed failed for {dst_tif} (world={wld}): {err}")
                else:
                    embedded_ok = True

            # Copy sidecars, but skip worldfile if embed succeeded
            for f in iter_sidecars(abs_p):
                if not f.is_file() or f in copied:
                    continue
                if embedded_ok and wld is not None and f.resolve(strict=False) == wld.resolve(strict=False):
                    copied.add(f)
                    continue
                dst_f = dst_for_src(f, src_root, dst_root)
                if dst_f is None:
                    continue
                msg = copy_file_quiet(f, dst_f, force=args.force, dry_run=args.dry_run)
                if msg:
                    failures.append(msg)
                copied.add(f)

            continue

        # Default: copy file + sidecars (quiet)
        for f in iter_sidecars(abs_p):
            if not f.is_file() or f in copied:
                continue
            dst_f = dst_for_src(f, src_root, dst_root)
            if dst_f is None:
                continue
            msg = copy_file_quiet(f, dst_f, force=args.force, dry_run=args.dry_run)
            if msg:
                failures.append(msg)
            copied.add(f)

    # Project rewrite last
    rewrite_project(
        qgs_path=qgs,
        original_text=qgs_text,
        srcs=srcs,
        dst_arg=args.dst_dir,
        dst_abs=dst_root,
        cwd=cwd,
        rewrites=rewrites,
        dry_run=args.dry_run,
        failures=failures,
    )

    # Report failures only
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f" - {f}")
        print(f"\nSummary: {len(failures)} failure(s)")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
