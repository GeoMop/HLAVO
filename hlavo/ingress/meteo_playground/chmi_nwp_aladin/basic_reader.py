from __future__ import annotations
from typing import *


from pathlib import Path

import bz2
import tempfile

import xarray as xr


def _read_grib_inner(
    grib_path: Path
) -> Tuple[Dict[str, Any], xr.Dataset]:
    """
    Inner reader: read an uncompressed GRIB file and return (metadata, xarray Dataset).
    - Engine fixed to cfgrib
    - Always loads into memory (no lazy)
    """

    backend_kwargs = {"indexpath": ""}  # don't write *.idx
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
    ds = ds.load()  # files are small => always load

    meta: Dict[str, Any] = {
        "grib_path": str(grib_path),
        "dims": dict(ds.dims),
        "coords": list(ds.coords),
        "data_vars": list(ds.data_vars),
        "attrs": dict(ds.attrs),
        "variables": {
            v: {
                "dims": ds[v].dims,
                "shape": list(ds[v].shape),
                "dtype": str(ds[v].dtype),
                "attrs": dict(ds[v].attrs),
            }
            for v in ds.data_vars
        },
    }

    return meta, ds


def read_grib_bz2(
    grib_bz2_path: Path,
    *,
    keep_temp: bool = False,
) -> Tuple[Dict[str, Any], xr.Dataset]:
    """
    Wrapper: decompress a .bz2 GRIB into a temp file, read it, cleanup (unless keep_temp=True).
    Returns (metadata, xarray Dataset).
    """
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="grib_"))
    out_name = grib_bz2_path.name
    if out_name.endswith(".bz2"):
        out_name = out_name[:-4]
    if not out_name.endswith((".grib", ".grb", ".grib2")):
        out_name += ".grib"
    tmp_grib_path = tmp_dir / out_name

    # Decompress bz2 -> temp GRIB
    with bz2.open(grib_bz2_path, "rb") as f_in, open(tmp_grib_path, "wb") as f_out:
        f_out.write(f_in.read())

    meta, ds = _read_grib_inner(tmp_grib_path)

    # Add wrapper-level info
    meta["source_file"] = str(grib_bz2_path)
    meta["decompressed_grib"] = str(tmp_grib_path)

    if not keep_temp:
        # ds is fully loaded => safe to delete temp file now
        try:
            tmp_grib_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
        meta.pop("decompressed_grib", None)

    return meta, ds


def report_grib_metadata(meta: Dict[str, Any], *, max_var_attrs: int = 25) -> None:
    """
    Pretty-print the metadata dictionary produced by read_bz2_grib().
    """
    print("GRIB METADATA REPORT")
    print("=" * 80)

    print(f"Source file: {meta.get('source_file')}")
    if "decompressed_grib" in meta:
        print(f"Decompressed: {meta.get('decompressed_grib')}")
    print(f"Engine:       {meta.get('engine')}")

    print("\nDataset")
    print("-" * 80)
    print(f"Dims:   {meta.get('dims')}")
    print(f"Coords: {meta.get('coords')}")
    print(f"Vars:   {meta.get('data_vars')}")

    attrs = meta.get("attrs", {}) or {}
    if attrs:
        print("\nGlobal attributes (top-level)")
        print("-" * 80)
        for k in sorted(attrs.keys()):
            v = attrs[k]
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "…"
            print(f"{k}: {s}")

    grib_summary = meta.get("grib_summary", {}) or {}
    if grib_summary:
        print("\nGRIB variable summary")
        print("-" * 80)
        for var, summ in grib_summary.items():
            print(f"* {var}")
            for k, v in summ.items():
                print(f"  - {k}: {v}")

    variables = meta.get("variables", {}) or {}
    if variables:
        print("\nVariables (dims/shape/dtype + attrs)")
        print("-" * 80)
        for var, info in variables.items():
            print(f"\n{var}")
            print(f"  dims:  {info.get('dims')}")
            print(f"  shape: {info.get('shape')}")
            print(f"  dtype: {info.get('dtype')}")

            vattrs = info.get("attrs", {}) or {}
            if vattrs:
                # show a bounded number of attrs to avoid walls of text
                keys = list(sorted(vattrs.keys()))[:max_var_attrs]
                print(f"  attrs (showing up to {max_var_attrs}):")
                for k in keys:
                    s = str(vattrs[k])
                    if len(s) > 200:
                        s = s[:200] + "…"
                    print(f"    {k}: {s}")



if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} <file.grib.bz2>", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])

    meta, ds = read_grib_bz2(path, keep_temp=False)

    # Print metadata as JSON to stdout
    print(json.dumps(meta, indent=2, default=str))

    # Quick dataset summary to stderr (so JSON output stays clean)
    print("\nDataset summary:", file=sys.stderr)
    print(ds, file=sys.stderr)