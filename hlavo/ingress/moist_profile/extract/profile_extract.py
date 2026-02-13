from pathlib import Path
from datetime import datetime
from typing import Any

import polars as pl
import re


# Matches "...:U30__51F3:1769975777477.csv" -> sensor_id = "U30"
_SENSOR_ID_RE = re.compile(r":(?P<sensor_id>[^:_]+)__")

# the positions of sensors on a probe [cm]
SENSOR_DEPTHS = [0.1, 0.2, 0.4, 0.6, 1.0]

# RENAME COLUMNS MAP
RENAME_MAP: dict[str, str] = {
    # "s2": "",
    # "s3": "",
    # "ttlDate": "",
    "s11t": "T_probe",
    # "s4": "",
    # "s5": "",
    "loggerUid": "manufacture_id",
    # "s4t": "",
    # "s3t": "",
    # "s5t": "",
    # "s2t": "",
    # "s1t": "",
    # "s1": "",
    # "logDateTime": "",
    "dateTime": "date_time",
}

# Optional helper if you prefer to only rename a subset:
# (You can delete keys you don't want to rename, or leave values as "" and they’ll be ignored.)


# DROP COLUMNS
DROP_COLS: set[str] = {
    "ttlDate",
    "logDateTime",
}

def _parse_time(time_str: str | None):
    if not time_str:
        return None
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except Exception:
        return time_str


def drop_columns(df: pl.DataFrame, cols_to_drop: set[str] | list[str] | tuple[str, ...]) -> pl.DataFrame:
    """
    Drop columns from a Polars DataFrame.
    - Ignores columns that don't exist.
    """
    cols = list(cols_to_drop)
    present = [c for c in cols if c in df.columns]
    return df.drop(present) if present else df


def rename_columns(df: pl.DataFrame, rename_map: dict[str, str]) -> pl.DataFrame:
    """
    Rename columns in a Polars DataFrame.
    - Renames only columns that exist AND have a non-empty new name.
    - Checks for target name collisions.
    """
    effective = {old: new for old, new in rename_map.items() if new and old in df.columns}

    targets = list(effective.values())
    if len(targets) != len(set(targets)):
        dupes = {t for t in targets if targets.count(t) > 1}
        raise ValueError(f"Rename target collision(s): {sorted(dupes)}")

    return df.rename(effective) if effective else df


# def reshape_moisture_long_explode(df: pl.DataFrame) -> pl.DataFrame:
#     moisture_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
#     if not moisture_cols:
#         raise ValueError("No moisture columns found matching s<digits> (e.g. s1, s2, ...).")
#
#     depth_levels = [int(c[1:]) for c in moisture_cols]
#     id_cols = [c for c in df.columns if c not in moisture_cols]
#
#     return (
#         df.select(
#             *[pl.col(c) for c in id_cols],
#             pl.concat_list([pl.col(c) for c in moisture_cols]).alias("moisture"),
#         )
#         .explode("moisture")
#         .with_columns(
#             # repeat depth levels for each row and align with exploded moisture
#             pl.int_range(0, pl.len()).mod(len(depth_levels)).map_elements(
#                 lambda idx: depth_levels[idx],
#                 return_dtype=pl.Int32,
#             ).alias("depth_level")
#         )
#     )


def reshape_columns(
    df: pl.DataFrame,
    *,
    moisture_col: str = "moisture",
    t_sensor_col: str = "T_sensor",
    depth_col: str = "depth_level",
) -> pl.DataFrame:
    """
    Reshape wide columns:
      - s{i}  -> moisture
      - s{i}t -> T_probe
    into long form with depth_level = i.

    Uses regex to find both column groups and asserts they match in length
    and depth indices.
    """

    # --- discover columns via regex ---
    moisture_cols = [c for c in df.columns if re.fullmatch(r"s\d", c)]
    temp_cols = [c for c in df.columns if re.fullmatch(r"s\dt", c)]

    if not moisture_cols:
        raise ValueError("No moisture columns found matching s<digits> (e.g. s1, s2, ...).")
    if not temp_cols:
        raise ValueError("No temperature columns found matching s<digits>t (e.g. s1t, s2t, ...).")

    # --- sort by depth index ---
    def depth_from_moist(c: str) -> int:
        return int(c[1:])              # "s12" -> 12

    def depth_from_temp(c: str) -> int:
        return int(c[1:-1])            # "s12t" -> 12

    moisture_cols = sorted(moisture_cols, key=depth_from_moist)
    temp_cols = sorted(temp_cols, key=depth_from_temp)

    moisture_depths = [depth_from_moist(c) for c in moisture_cols]
    temp_depths = [depth_from_temp(c) for c in temp_cols]

    assert len(moisture_cols) == len(temp_cols), (
        f"Different number of moisture vs temp columns: "
        f"{len(moisture_cols)} vs {len(temp_cols)}"
    )
    assert moisture_depths == temp_depths, (
        f"Depth indices do not match:\n"
        f"moisture depths: {moisture_depths}\n"
        f"temp depths:     {temp_depths}"
    )

    depth_levels = moisture_depths
    k = len(depth_levels)

    # --- id columns: everything except s{i} and s{i}t ---
    exclude = set(moisture_cols) | set(temp_cols)
    id_cols = [c for c in df.columns if c not in exclude]

    # --- reshape ---
    depth_list_expr = pl.lit(depth_levels).cast(pl.List(pl.Int8))
    idx_expr = pl.int_range(0, pl.len()).mod(k)

    return (
        df.select(
            *[pl.col(c) for c in id_cols],
            pl.concat_list([pl.col(c) for c in moisture_cols]).alias(moisture_col),
            pl.concat_list([pl.col(c) for c in temp_cols]).alias(t_sensor_col),
        )
        .explode([moisture_col, t_sensor_col])  # explode in parallel
        .with_columns(depth_list_expr.list.get(idx_expr).alias(depth_col))
    )


def add_sensor_depth(df: pl.DataFrame, depths: list[float], *, depth_col: str = "depth_level") -> pl.DataFrame:
    """
    Add sensor_depth based on depth_level (1-based index).
    depths[0] corresponds to depth_level=1, etc.
    """
    mapping = {i + 1: d for i, d in enumerate(depths)}  # 1->10, 2->20, ...

    return df.with_columns(
        pl.col(depth_col).map_elements(lambda x: mapping.get(int(x)),
                                       return_dtype=pl.Float64).alias("sensor_depth")
    )


def load_locations_csv(
    path: str,
    *,
    date_col: str = "date",
    time_col: str = "time",
    uid_col: str = "loggerUid",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    dt_col_out: str = "loc_datetime",
) -> pl.DataFrame:
    """
    Reads locations CSV like:
      date,time,loggerUid,latitude,longitude
      2025-04-29,11:32:16,U01,50.863565,14.889853

    Returns a DataFrame with columns:
      loggerUid, loc_datetime, latitude, longitude
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


def add_location_to_measurements(
    measurements: pl.DataFrame,
    locations: pl.DataFrame,
    *,
    meas_uid_col: str = "probe_id",
    meas_dt_col: str = "date_time",
    loc_uid_col: str = "loggerUid",
    loc_dt_col: str = "loc_datetime",
) -> pl.DataFrame:
    """
    Adds latitude/longitude to each measurement row based on:
    - same loggerUid
    - latest location timestamp <= measurement timestamp
    """
    # join_asof requires both sides sorted by the join keys
    meas_sorted = measurements.sort([meas_uid_col, meas_dt_col])
    loc_sorted = locations.sort([loc_uid_col, loc_dt_col])

    return meas_sorted.join_asof(
        loc_sorted,
        left_on=meas_dt_col,
        right_on=loc_dt_col,
        by_left=meas_uid_col,
        by_right=loc_uid_col,
        strategy="backward",  # last known location at/before measurement time
    ).drop(loc_dt_col)  # optional: drop loc_datetime after join


def extract_df(file_path: str | Path, *, read_csv_kwargs: dict | None = None) -> pl.DataFrame:
    """
    Read a single CSV file into a Polars DataFrame, extract sensor_id from the filename,
    and add it as a new column.
    """
    p = Path(file_path)
    name = p.name

    m = _SENSOR_ID_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match expected format for sensor id: {name}")

    sensor_id = m.group("sensor_id")

    kwargs = read_csv_kwargs or {}
    # dateTime: 06-01-2025 11:00:00
    df = pl.read_csv(p, **kwargs)
    df = df.with_columns( pl.col("dateTime")
        .str.strptime(pl.Datetime, format="%d-%m-%Y %H:%M:%S", strict=True)
        .alias("dateTime")
    )

    df = drop_columns(df, DROP_COLS)
    df = rename_columns(df, RENAME_MAP)
    df = df.with_columns(pl.lit(sensor_id).alias("probe_id"))
    # df = reshape_moisture_long_explode(df)
    df = reshape_columns(df)
    df = add_sensor_depth(df, SENSOR_DEPTHS)

    # add latitude, longitude
    loc = load_locations_csv("location_in_time.csv")
    # print(loc)
    df = add_location_to_measurements(df, loc)

    return df
