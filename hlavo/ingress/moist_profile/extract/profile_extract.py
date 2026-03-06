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


def add_location_to_measurements(
    df_meas: pl.DataFrame,
    df_sites: pl.DataFrame
) -> pl.DataFrame:
    """
        Add site_id/latitude/longitude onto measurements using df_sites as an assignment timeline.

        Behavior:
        - When a probe_id (U*) is assigned to a site, the mapping applies forward in time
        - When an 'X' is found for that site, it UNASSIGNS the currently assigned probe
          from that moment onward (including that timestamp)
    """

    # 1) For rows where df_sites.probe_id == "X", determine which probe it cancels:
    #    it's the last seen non-X probe_id in that site up to that time.
    sites = df_sites.sort(["site_id", "datetime"]).with_columns(
        active_probe=(
            pl.when((pl.col("probe_id").is_not_null()) & (pl.col("probe_id") != "X"))
            .then(pl.col("probe_id"))
            .otherwise(None)
            .forward_fill()
            .over("site_id")
        )
    )

    # 2) Build probe event timeline:
    #    - assign events for U* rows (event_probe = probe_id)
    #    - unassign events for X rows (event_probe = active_probe)
    assign_events = (
        sites
        .filter(pl.col("probe_id").is_not_null() & (pl.col("probe_id") != "X"))
        .select(
            pl.col("datetime").alias("event_dt"),
            pl.col("probe_id").alias("event_probe"),
            "site_id", "latitude", "longitude",
            pl.lit(False).alias("is_unassign"),
        )
    )

    unassign_events = (
        sites
        .filter((pl.col("probe_id") == "X") & pl.col("active_probe").is_not_null())
        .select(
            pl.col("datetime").alias("event_dt"),
            pl.col("active_probe").alias("event_probe"),
            pl.lit(None).cast(pl.Int64).alias("site_id"),
            pl.lit(None).cast(pl.Float64).alias("latitude"),
            pl.lit(None).cast(pl.Float64).alias("longitude"),
            pl.lit(True).alias("is_unassign"),
        )
    )

    events = (
        pl.concat([assign_events, unassign_events], how="vertical")
        # If there are multiple events with same (probe, time),
        # make sure UNASSIGN wins for exact matches.
        .with_columns(priority=pl.when(pl.col("is_unassign")).then(2).otherwise(1))
        .sort(["event_probe", "event_dt", "priority"])
        .unique(subset=["event_probe", "event_dt"], keep="last")
        .drop("priority")
        .sort(["event_probe", "event_dt"])
    )

    # 3) Asof-join measurements to the latest event for that probe
    meas_sorted = df_meas.sort(["probe_id", "date_time"])

    out = (
        meas_sorted.join_asof(
            events,
            left_on="date_time",
            right_on="event_dt",
            by_left="probe_id",
            by_right="event_probe",
            strategy="backward",
        )
        # set zero site_id wherever null - means probe is not used
        # .with_columns(site_id=pl.col("site_id").cast(pl.Int64).fill_null(0))
        # keep only measurements assigned to a site
        .filter(pl.col("site_id").is_not_null())
        # drop auxiliary columns
        .drop(["event_dt", "is_unassign"])
    )

    return out


def extract_df(file_path: str | Path, df_sites,
               *, read_csv_kwargs: dict | None = None) -> pl.DataFrame | None:
    """
    Read a single CSV file into a Polars DataFrame, extract sensor_id from the filename,
    and add it as a new column.
    Returns None if no data found in CSV (no dateTime column).
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

    # test due to empty files (no data):
    if "dateTime" not in df.columns:
        return None

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

    # add probe type column with constant values "Odyssey"
    df = df.with_columns(pl.lit("Odyssey").alias("probe_model"))
    # add permeability column with zeros
    df = df.with_columns(pl.lit(0.0).alias("permeability"))

    # add latitude, longitude
    df = add_location_to_measurements(df, df_sites)
    df = df.with_columns(pl.lit(0).alias("site_status"))

    return df
