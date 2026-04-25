from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Iterator

import zarr_fuse as zf

from hlavo.misc.aux_zarr_fuse import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_PATH = REPO_ROOT / "hlavo" / "schemas"


def _iter_schema_paths(schema_dir: Path) -> list[Path]:
    return sorted(schema_dir.glob("*.yaml"))


def _store_name(attrs: dict) -> str:
    store_url = attrs["STORE_URL"] if "STORE_URL" in attrs else attrs["S3_STORE_URL"]
    store_url = str(store_url)
    return Path(store_url.removeprefix("s3://").removeprefix("file://")).name


def _matches_path_glob(*, full_path: str, path_glob: str) -> bool:
    return fnmatch.fnmatch(full_path, path_glob)


def _matches_store_prefix(*, store_name: str, path_glob: str) -> bool:
    return (
        _matches_path_glob(full_path=store_name, path_glob=path_glob)
        or _matches_path_glob(full_path=f"{store_name}/*", path_glob=path_glob)
        or path_glob.startswith(f"{store_name}/")
    )


def _format_group_line(*, full_path: str, node) -> str:
    child_count = len(node.children)
    array_count = len(node.dataset.data_vars) + len(node.dataset.coords)
    return f"  {full_path} [group] groups={child_count} arrays={array_count}"


def _iter_dataset_lines(*, node) -> Iterator[str]:
    return (f"    {line}" for line in str(node.dataset).splitlines())


def _iter_matching_nodes(*, node, store_name: str, path_glob: str) -> Iterator[tuple[str, object]]:
    node_path = node.group_path
    full_path = store_name if node_path == "" else f"{store_name}/{node_path}"

    if _matches_path_glob(full_path=full_path, path_glob=path_glob):
        yield full_path, node

    for _, child in sorted(node.items()):
        yield from _iter_matching_nodes(
            node=child,
            store_name=store_name,
            path_glob=path_glob,
        )


def run_dataset_cli(
    *,
    path_glob: str = "*",
    schema_dir: Path = SCHEMAS_PATH,
    print_dataset: bool = False,
) -> int:
    load_dotenv()
    schema_paths = _iter_schema_paths(schema_dir)
    assert schema_paths, f"No schema files found in {schema_dir}"

    for schema_path in schema_paths:
        schema = zf.schema.deserialize(schema_path)
        schema_attrs = schema.ds.ATTRS
        store_name = _store_name(schema_attrs)
        if not _matches_store_prefix(store_name=store_name, path_glob=path_glob):
            continue

        print()
        print(f"store: {store_name}")
        try:
            root = zf.open_store(schema)
            for full_path, node in _iter_matching_nodes(
                node=root,
                store_name=store_name,
                path_glob=path_glob,
            ):
                print(_format_group_line(full_path=full_path, node=node))
                if print_dataset:
                    for line in _iter_dataset_lines(node=node):
                        print(line)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect HLAVO zarr_fuse stores listed in hlavo/schemas.",
    )
    parser.add_argument(
        "path_glob",
        nargs="?",
        default="*",
        help='Optional full-path glob like "wells.zarr/Uhelna/*". Default is "*".',
    )
    parser.add_argument(
        "-p",
        "--print-dataset",
        action="store_true",
        help="Print full xarray dataset repr for matched nodes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_dataset_cli(path_glob=args.path_glob, print_dataset=args.print_dataset)


if __name__ == "__main__":
    raise SystemExit(main())
