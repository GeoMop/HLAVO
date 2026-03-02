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


def read_csv_files_to_polars(filepaths: list[str | Path], *, read_csv_kwargs: dict | None = None)\
        -> pl.DataFrame | None:
    """
    Process a list of CSV filepaths and concatenate into one Polars DataFrame.
    """
    paths = [Path(p) for p in filepaths]
    if not paths:
        return pl.DataFrame()

    dfs: list[pl.DataFrame] = []
    for p in paths:
        df = extract_df(p, read_csv_kwargs=read_csv_kwargs)
        if df is not None:
            dfs.append(df)

    # If columns differ across files, consider: how="diagonal"
    if len(dfs) == 0:
        return None
    else:
        return pl.concat(dfs, how="vertical", rechunk=True)


def override_local_storage(schema, storage_path: str | Path | None):
    if storage_path is not None:
        schema.ds.ATTRS['STORE_URL'] = str(storage_path)


def main(source_dir: str | Path, storage_path: str | Path = None) -> None:
    """
    :param source_dir: source directory with CSV files from xpert.nz datareports
    :return:
    """
    load_dotenv("../.env")
    schema_path = Path("../profile_schema.yaml")

    csv_files = list_csv_filepaths(source_dir)
    df = read_csv_files_to_polars(csv_files)
    if df is None:
        print(f"No data found in given directory: {source_dir}")
        return None
    print(df)

    schema = zarr_fuse.schema.deserialize(schema_path)
    override_local_storage(schema, storage_path)


    root_node = zarr_fuse.open_store(schema)
    print('Store open')
    root_node['Uhelna']['profiles'].update(df)
    print('Updated')
    rdf = root_node['Uhelna']['profiles'].read_df(var_names=["moisture"])
    print(rdf)

    # works locally
    # test on S3
    # zarr_fuse.remove_store(schema, workdir=workdir)


if __name__ == '__main__':
    root_path = Path(__file__).parents[4]
    # main(source_dir=root_path / "tests/ingress/moist_profile/20260201T205548_dataflow_grab")
    # 2025 01-06
    main(source_dir="../20260301T224908_dataflow_grab",
         storage_path=Path("storage_2025"))
    # 2025 07-12
    main(source_dir="../20260301T225923_dataflow_grab",
         storage_path=Path("storage_2025"))
    # main(source_dir="../test_empty",
    #      storage_path=Path("storage_empty"))