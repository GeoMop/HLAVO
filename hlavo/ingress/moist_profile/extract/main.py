# from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
import polars as pl
from profile_extract import extract_df
import zarr_fuse


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


def main(source_dir: str | Path) -> None:
    """
    :param source_dir: source directory with CSV files from xpert.nz datareports
    :return:
    """
    load_dotenv("../.env")
    schema_path = Path("../profile_schema.yaml")

    csv_files = list_csv_filepaths(source_dir)
    df = read_csv_files_to_polars(csv_files)
    print(df)

    schema = zarr_fuse.schema.deserialize(schema_path)
    root_node = zarr_fuse.open_store(schema)
    print('Store open')
    root_node['Uhelna']['profiles'].update(df)
    print('Updated')
    # rdf = root_node['Uhelna']['profiles'].read_df(var_names=["moisture"])
    # print(rdf)

    # works locally
    # test on S3
    # zarr_fuse.remove_store(schema, workdir=workdir)


if __name__ == '__main__':
    root_path = Path(__file__).parents[4]
    main(source_dir=root_path / "tests/ingress/moist_profile/20260201T205548_dataflow_grab")