# from __future__ import annotations
from pathlib import Path
import polars as pl
from profile_extract import extract_df


def list_csv_filepaths(dir_path: str | Path) -> list[Path]:
    """
    List CSV filepaths in a directory (sorted).
    """
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    return sorted(d.glob("*.csv"))


def read_csv_files_to_polars(filepaths: list[str | Path], *, read_csv_kwargs: dict | None = None) -> pl.DataFrame:
    """
    Process a list of CSV filepaths and concatenate into one Polars DataFrame.
    """
    paths = [Path(p) for p in filepaths]
    if not paths:
        return pl.DataFrame()

    dfs: list[pl.DataFrame] = []
    for p in paths:
        dfs.append(extract_df(p, read_csv_kwargs=read_csv_kwargs))

    # If columns differ across files, consider: how="diagonal"
    return pl.concat(dfs, how="vertical", rechunk=True)


# script: zavola zarr_fuse, vytvori lokalni zarr storage (cesta z dict parametru)
def main():
    source_dir = "../20260201T205548_dataflow_grab"
    csv_files = list_csv_filepaths(source_dir)
    df = read_csv_files_to_polars(csv_files)
    print(df)


if __name__ == '__main__':
    main()