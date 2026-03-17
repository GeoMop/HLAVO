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


def read_csv_files_to_polars(filepaths: list[str | Path], df_sites: pl.DataFrame,
                             *, read_csv_kwargs: dict | None = None)\
        -> list[pl.DataFrame] | None:
    """
    Process a list of CSV filepaths and concatenate into one Polars DataFrame.
    """
    paths = [Path(p) for p in filepaths]
    if not paths:
        return None

    dfs: list[pl.DataFrame] = []
    for p in paths:
        print(f"Processing {p}")
        df = extract_df(p, df_sites, read_csv_kwargs=read_csv_kwargs)
        if df is not None and not df.is_empty():
            dfs.append(df)

    # If columns differ across files, consider: how="diagonal"
    if len(dfs) == 0:
        return None
    else:
        return dfs
        # return pl.concat(dfs, how="vertical", rechunk=True)


def load_site_coords_csv(
    path: str,
    *,
    date_col: str = "date",
    time_col: str = "time",
    uid_col: str = "site_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    dt_col_out: str = "site_datetime",
) -> pl.DataFrame:
    """
    Reads locations CSV like:
      date,time,site_id,latitude,longitude
      2025-04-29,11:32:16,1,50.863565,14.889853

    Returns a DataFrame with columns:
      site_id, site_datetime, latitude, longitude
    """
    df = pl.read_csv(path)

    df = df.with_columns(
        # build a single datetime column
        pl.concat_str([pl.col(date_col), pl.col(time_col)], separator=" ")
          .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=True)
          .alias(dt_col_out),

        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ).select([uid_col, dt_col_out, lat_col, lon_col])

    # join_asof requires sorting
    return df.sort([uid_col, dt_col_out])


def load_site_status_csv(
    path: str,
    *,
    date_col: str = "date",
    time_col: str = "time",
    dt_col_out: str = "status_datetime",
) -> pl.DataFrame:
    """
    Reads locations CSV like:
      date,time,1,2,3,4,5,...
      2025-04-29,11:32:16,1,50.863565,14.889853

    Returns a DataFrame with columns:
      status_datetime, 1,2,3,4,5,...
    """
    df = pl.read_csv(path, skip_lines=20)

    # Build a clean time column:
    #   - null/empty -> 0:00:00
    #   - HH:MM      -> HH:MM:00
    #   - HH:MM:SS   -> keep as-is
    time_fixed = (
        pl.when(pl.col(time_col).is_null() | (pl.col(time_col).cast(pl.Utf8).str.strip_chars() == ""))
        .then(pl.lit("0:00"))
        # .when(pl.col(time_col).cast(pl.Utf8).str.count_matches(":") == 1)
        # .then(pl.concat_str([pl.col(time_col).cast(pl.Utf8), pl.lit(":00")]))
        .otherwise(pl.col(time_col).cast(pl.Utf8))
        .alias(time_col)
    )
    df = df.with_columns(time_fixed)

    df = df.with_columns(
        # build a single datetime column
        pl.concat_str([pl.col(date_col), pl.col(time_col)], separator=" ")
          .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M", strict=True)
          .alias(dt_col_out),

    )

    df = df.drop([date_col, time_col, "author", "note"])
    return df.sort([dt_col_out])


def status_to_site_frames(
    df_coords: pl.DataFrame,
    df_status: pl.DataFrame,
    *,
    site_id_col: str = "site_id",
    site_dt_col: str = "site_datetime",
    status_dt_col: str = "status_datetime",
) -> pl.DataFrame:
    """
    For each status column (header is an integer site_id), produce a DF with:
      datetime, probe_id, site_id, latitude, longitude

    Rules:
    - probe_id is the last seen string starting with "U" (e.g., U04)
    - probe_id stays valid for later datetimes until a new "U*" appears
    - if "X" or null appears, probe_id becomes null from that row onward
      until a later "U*" appears
    - latitude/longitude are taken from df_coords using an asof (backward) join
      per site_id on datetime
    """

    # --- 1) Wide -> long (one row per datetime+site_id)
    site_cols = [c for c in df_status.columns if c != status_dt_col]

    long = (
        df_status
        .unpivot(
            index=status_dt_col,
            on=site_cols,
            variable_name=site_id_col,
            value_name="raw",
        )
        .with_columns(
            pl.col(site_id_col).cast(pl.Int64),
            pl.col("raw").cast(pl.Utf8),
        )
        .rename({status_dt_col: "datetime"})
    )

    # --- 2) Sticky probe_id with resets on X/null, and updates on U*
    raw = pl.col("raw")
    # Only X breaks the carry-forward
    is_x = raw.eq("X").fill_null(False)  # bool, never null
    segment = is_x.cast(pl.Int64).cum_sum().over("site_id")  # int, never null

    probe_marker = pl.when(raw.str.starts_with("U").fill_null(False)).then(raw).otherwise(None)

    long = (
        long
        .with_columns(raw=raw)  # ensure consistent dtype
        .with_columns(segment=segment)
        .with_columns(
            probe_id=probe_marker.forward_fill().over(["site_id", "segment"]),
        )
        .with_columns(
            probe_id=pl.when(is_x).then(pl.lit("X")).otherwise(pl.col("probe_id")),
        )
    )

    # -----------------------------
    # 3) site_status
    # -----------------------------
    # parse signed integers from raw; non-integers become null
    raw_int = raw.cast(pl.Int64, strict=False)

    is_probe = raw.str.starts_with("U").fill_null(False)
    is_int = raw_int.is_not_null()

    # adjust this if you actually want >= 10 to persist
    is_persistent_status = is_int & (raw_int >= 10)
    is_transient_status = is_int & (raw_int < 10)

    # look at the next row within each site
    next_raw_int = raw_int.shift(-1).over(site_id_col)
    next_is_persistent = next_raw_int.is_not_null() & (next_raw_int >= 10)

    # any explicit value/probe/X breaks previous persistent carry
    status_boundary = (is_probe | is_x | is_int)

    status_segment = status_boundary.cast(pl.Int64).cum_sum().over(site_id_col)

    persistent_status = (
        pl.when(is_persistent_status)
        .then(raw_int)
        .otherwise(None)
        .forward_fill()
        .over([site_id_col, status_segment])
    )

    # long = (
    #     long
    #     .with_columns(
    #         raw_int=raw_int,
    #         status_segment=status_segment,
    #         site_status=pl.when(is_transient_status)
    #         .then(raw_int)  # only current row
    #         .otherwise(persistent_status),  # carried over nulls
    #     )
    # )
    long = (
        long
        .with_columns(
            raw_int=raw_int,
            next_raw_int=next_raw_int,
            status_segment=status_segment,
        )
        .with_columns(
            site_status=
            pl.when(is_probe & next_is_persistent)
            .then(next_raw_int)  # U* row gets next row's persistent status
            .when(is_transient_status)
            .then(raw_int)  # <10 only on current row
            .otherwise(persistent_status)  # >=10 persists forward
        )
    )

    # --- 3) Asof-join latitude/longitude from df_coords (closest previous site_datetime)
    site_sorted = df_coords.sort([site_id_col, site_dt_col])
    long_sorted = long.sort([site_id_col, "datetime"])
    # long_pd = long_sorted.to_pandas()

    joined = (
        long_sorted.join_asof(
            site_sorted,
            left_on="datetime",
            right_on=site_dt_col,
            by=site_id_col,
            strategy="backward",
        )
        .select(["datetime", "probe_id", site_id_col, "latitude", "longitude", "site_status"])
    )
    print(joined)
    # joined_np = joined.to_pandas()

    # --- 4) Return list of per-site dataframes
    # return joined.partition_by(site_id_col, as_dict=False, maintain_order=True)
    return joined


def split_lab(dfs: list[pl.DataFrame]) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """
    Split a list of dataframes into:
      - one dataframe containing all rows with site_id == 5
      - list of dataframes containing the remaining rows
    """

    lab_parts = []
    rest_dfs = []

    for df in dfs:
        mask = pl.col("site_id") == 5

        site_lab = df.filter(mask)
        rest = df.filter(~mask)

        if not site_lab.is_empty():
            lab_parts.append(site_lab)

        if not rest.is_empty():
            rest_dfs.append(rest)

    df_lab = pl.concat(lab_parts) if lab_parts else pl.DataFrame()

    return df_lab, rest_dfs


def override_local_storage(schema, storage_path: str | Path | None):
    if storage_path is not None:
        schema.ds.ATTRS['STORE_URL'] = str(storage_path)


def load_schema():
    load_dotenv("../.env")
    schema_path = Path("../profile_schema.yaml")
    schema = zarr_fuse.schema.deserialize(schema_path)
    return schema


def main(source_dir: str | Path, storage_path: str | Path = None) -> None:
    """
    :param source_dir: source directory with CSV files from xpert.nz datareports
    :return:
    """
    schema = load_schema()
    override_local_storage(schema, storage_path)

    df_coords = load_site_coords_csv("site_coords.csv")
    print(df_coords)
    df_status = load_site_status_csv("site_status.csv")
    print(df_status)

    df_sites = status_to_site_frames(df_coords, df_status)
    print(df_sites)

    csv_files = list_csv_filepaths(source_dir)
    dfs = read_csv_files_to_polars(csv_files, df_sites)
    if dfs is None:
        print(f"No data found in given directory: {source_dir}")
        return None
    print(dfs)

    df_lab, dfs_network = split_lab(dfs)

    root_node = zarr_fuse.open_store(schema)
    print('Store open')
    for i, df in enumerate(dfs_network):
        print(f'Storing df[{i}/{len(dfs_network)}]: {df.shape[0]} rows.')
        root_node['Uhelna']['profiles'].update(df)
    print('Updated')

    if not df_lab.is_empty():
        print(f'Storing lab df: {df_lab.shape[0]} rows.')
        root_node['Uhelna']['lab'].update(df_lab)

    # close/open again
    # root_node = zarr_fuse.open_store(schema)
    # rdf = root_node['Uhelna']['profiles'].read_df(var_names=["moisture", "probe_id"])
    # print(rdf)

    # works locally
    # test on S3
    # zarr_fuse.remove_store(schema, workdir=workdir)
    pass


def read_storage(storage_path: str | Path = None):
    schema = load_schema()
    override_local_storage(schema, storage_path)

    root_node = zarr_fuse.open_store(schema)
    rdf = root_node['Uhelna']['profiles'].read_df(var_names=["moisture", "probe_id", "sensor_depth"])
    print(rdf)

    # place debug pause here and view pandas dataframe
    rdf_pd = rdf.sort(["site_id", "date_time", "depth_level"]).to_pandas()
    pass


def remove_S3_storage():
    schema = load_schema()
    zarr_fuse.remove_store(schema)


if __name__ == '__main__':
    root_path = Path(__file__).parents[4]
    # main(source_dir=root_path / "tests/ingress/moist_profile/20260201T205548_dataflow_grab",
    #      storage_path=root_path / "tests/ingress/moist_profile/test_storage")
        # storage_path = "test_storage")
    # 2025 01-06
    # main(source_dir="../20260301T224908_dataflow_grab",
    #      storage_path=Path("storage_2025"))
    # 2025 07-12
    # main(source_dir="../20260301T225923_dataflow_grab",
    #      storage_path=Path("storage_2025"))
    # main(source_dir="../test_empty",
    #      storage_path=Path("storage_empty"))


    # TO S3
    # remove_S3_storage()
    # main(source_dir="../20260301T224908_dataflow_grab")
    # main(source_dir="../20260301T225923_dataflow_grab")
    # read_storage()

    # read to check zarr storage
    # read_storage(storage_path=Path("test_storage"))
    # read_storage(storage_path=root_path / "tests/ingress/moist_profile/test_storage")
    # read_storage(storage_path=Path("storage_2025"))
