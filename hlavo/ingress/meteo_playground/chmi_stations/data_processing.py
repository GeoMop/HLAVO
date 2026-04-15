import json
import logging
from pathlib import Path

import polars as pl
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from hlavo.common.zarr_fuse_reader import read_storage
from hlavo.ingress.meteo_playground.chmi_stations.config import (
    ACTIVE_STATIONS_CSV_PATH,
    CHMI_STATIONS_SCHEMA_PATH,
    CHMI_STATIONS_STORAGE_PATH,
    CLM_REQUIRED_SOURCE_VARS,
    CLM_STATION_PRIORITY,
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    NODE_CHMI_STATIONS,
    NODE_OPEN_METEO,
    NODE_PARFLOW,
    OPEN_METEO_DATA_HOURLY_PATH,
    OPEN_METEO_REQUIRED_VARS,
    PARFLOW_OPEN_METEO_COMPARISON_PLOT_PATH,
    SCRIPT_DIR,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SITE_COORDS_CSV_PATH,
    STATIONS_DATA_DAILY_PATH,
    STATIONS_DATA_HOURLY_PATH,
    STATION_DF_METADATA_COLUMNS,
)
from hlavo.ingress.moist_profile.extract.main import load_site_coords_csv
from hlavo.ingress.meteo_playground.chmi_stations.open_meteo import open_meteo
from hlavo.ingress.meteo_playground.chmi_stations.data_scrapper import get_station_coordinates


LOGGER = logging.getLogger(__name__)


def get_data_block(doc):
    """
    Return the dict that contains CHMI 'header' and 'values'.
    """
    if "header" in doc and "values" in doc:
        return doc

    data = doc.get("data")
    if isinstance(data, dict):
        if "header" in data and "values" in data:
            return data
        inner = data.get("data")
        if isinstance(inner, dict) and "header" in inner and "values" in inner:
            return inner

    raise RuntimeError("Could not find 'header'/'values' block in JSON document.")


def find_station_data_paths(wsi, stations_data_dir, pattern, *, require_single=False):
    """
    Find station JSON files under a directory tree, allowing nested year/month partitions.
    """
    stations_path = Path(stations_data_dir)
    matching_paths = sorted(stations_path.rglob(pattern.format(wsi=wsi)))
    if not matching_paths:
        raise FileNotFoundError(f"Station data files not found under {stations_path}: {pattern.format(wsi=wsi)}")
    if require_single:
        assert len(matching_paths) == 1, (
            f"Expected one match for {pattern.format(wsi=wsi)}, got {len(matching_paths)}."
        )
    return matching_paths


def load_station_dataframe_from_paths(station_paths, index_columns):
    """
    Read one or more CHMI station JSON files, merge them, and pivot ELEMENT into columns.
    """
    frames = []
    for station_path in station_paths:
        with station_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        data_block = get_data_block(doc)
        header = data_block["header"].split(",")
        values = data_block["values"]
        header_index = {column_name: index for index, column_name in enumerate(header)}
        required_columns = [*index_columns, "ELEMENT", "VAL"]

        frame = pl.DataFrame(
            [
                {
                    column_name: (
                        None
                        if row[header_index[column_name]] in ("", None)
                        else str(row[header_index[column_name]])
                    )
                    for column_name in required_columns
                }
                for row in values
            ],
            schema={column_name: pl.Utf8 for column_name in required_columns},
        ).with_columns(
            pl.col("DT").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False),
            pl.col("VAL").cast(pl.Utf8).str.strip_chars().replace("", None).cast(pl.Float64, strict=False),
        )
        frames.append(frame)

    df = pl.concat(frames, how="vertical")
    df = df.pivot(
        on="ELEMENT",
        values="VAL",
        index=index_columns,
        aggregate_function="first",
    )
    return df.sort("DT").rename({"DT": "date_time"})


def filter_date_interval(df, start_date=None, end_date=None):
    """
    Filter a dataframe by the inclusive date_time interval.
    """
    if start_date is not None:
        df = df.filter(pl.col("date_time") >= pl.lit(start_date).str.to_datetime(time_zone="UTC"))
    if end_date is not None:
        df = df.filter(pl.col("date_time") <= pl.lit(end_date).str.to_datetime(time_zone="UTC"))
    return df


def load_station_daily_dataframe(
    wsi,
    stations_data_dir=STATIONS_DATA_DAILY_PATH,
    start_date=None,
    end_date=None,
):
    """
    Read one downloaded CHMI daily data file, pivot ELEMENT values into columns,
    sort by datetime, and filter by the selected date interval.
    """
    station_paths = find_station_data_paths(
        wsi=wsi,
        stations_data_dir=stations_data_dir,
        pattern="dly-{wsi}.json",
        require_single=True,
    )
    df = load_station_dataframe_from_paths(station_paths, index_columns=["STATION", "VTYPE", "DT"])
    return filter_date_interval(
        df=df,
        start_date=start_date,
        end_date=end_date,
    )


def load_station_hourly_dataframe(
    wsi,
    stations_data_dir=STATIONS_DATA_HOURLY_PATH,
    start_date=None,
    end_date=None,
):
    """
    Read all downloaded CHMI hourly files for one station, merge through years/months,
    pivot ELEMENT values into columns, and filter by the selected date interval.
    """
    station_paths = find_station_data_paths(
        wsi=wsi,
        stations_data_dir=stations_data_dir,
        pattern="1h-{wsi}-*.json",
    )
    return filter_date_interval(
        df=load_station_dataframe_from_paths(station_paths, index_columns=["STATION", "DT"]),
        start_date=start_date,
        end_date=end_date,
    )


def validate_station_quantity_flags(station_df, station):
    """
    Compare station metadata flags against actual station_df content.
    """
    metadata_columns = {
        "WSI",
        "FULL_NAME",
        "LAT",
        "LON",
        "ELEVATION",
        "DIST_KM",
        "BEGIN_DATE",
        "END_DATE",
    }

    quantity_columns = [
        column_name
        for column_name in station_df.columns
        if column_name not in STATION_DF_METADATA_COLUMNS
    ]
    all_null_quantity_columns = {
        column_name
        for column_name in quantity_columns
        if not station_df.select(pl.col(column_name).is_not_null().any()).item()
    }
    columns_with_values = {
        column_name
        for column_name in quantity_columns
        if station_df.select(pl.col(column_name).is_not_null().any()).item()
    }
    false_quantity_columns = {
        column_name
        for column_name, value in station.items()
        if column_name not in metadata_columns and str(value).lower() == "false"
    }
    true_quantity_columns = {
        column_name
        for column_name, value in station.items()
        if column_name not in metadata_columns and str(value).lower() == "true"
    }

    validation = {
        "null_despite_true": sorted(all_null_quantity_columns & true_quantity_columns),
        "values_despite_false": sorted(columns_with_values & false_quantity_columns),
    }

    if validation["null_despite_true"]:
        LOGGER.warning(
            "Station %s has quantity columns flagged True but containing only null values: %s",
            station["WSI"],
            validation["null_despite_true"],
        )
    if validation["values_despite_false"]:
        LOGGER.warning(
            "Station %s has quantity columns flagged False but containing valid values: %s",
            station["WSI"],
            validation["values_despite_false"],
        )

    return validation


def drop_all_null_columns(df, excluded_columns):
    """
    Drop columns that contain only null values, except for required metadata columns.
    """
    keep_columns = [
        column_name
        for column_name in df.columns
        if column_name in excluded_columns or df.select(pl.col(column_name).is_not_null().any()).item()
    ]
    return df.select(keep_columns)


def print_station_columns(station_name, column_names, name_width):
    """
    Print station name with fixed width followed by the available quantity columns.
    """
    print(f"{station_name:<{name_width}} {column_names}")


# def get_node_quantity_columns(node):
#     """
#     Return quantity dataframe columns declared for the node schema.
#     """
#     return [
#         var_config.df_col
#         for var_config in node.schema.VARS.values()
#     ]
#
#
# def add_missing_schema_columns(stations_df, node):
#     """
#     Ensure the dataframe contains all quantity columns required by the node schema.
#     Missing quantity columns are added and filled with null values.
#     """
#     required_quantity_columns = get_node_quantity_columns(node)
#     missing_columns = [
#         column_name
#         for column_name in required_quantity_columns
#         if column_name not in stations_df.columns
#     ]
#
#     if not missing_columns:
#         return stations_df
#
#     return stations_df.with_columns(
#         [pl.lit(None, dtype=pl.Float64).alias(column_name) for column_name in missing_columns]
#     )


def load_active_station_dataframe(
    *,
    station_loader,
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    stations_data_dir,
    allow_missing=False,
    validate_flags=False,
    start_date=None,
    end_date=None,
):
    """
    Load station data for each active station, append coordinates, concatenate the
    result, and print the available quantity columns for visual inspection.
    """
    stations_df = pl.read_csv(stations_csv_path)
    station_name_width = stations_df.select(pl.col("FULL_NAME").str.len_chars().max()).item()

    station_frames = []
    for station in stations_df.iter_rows(named=True):
        try:
            station_df = station_loader(
                wsi=station["WSI"],
                stations_data_dir=stations_data_dir,
                start_date=start_date,
                end_date=end_date,
            )
        except FileNotFoundError:
            if not allow_missing:
                raise
            LOGGER.warning("Station data missing for %s (%s). Skipping.", station["FULL_NAME"], station["WSI"])
            continue

        station_df = station_df.with_columns(
            pl.lit(station["LAT"]).cast(pl.Float64).alias("latitude"),
            pl.lit(station["LON"]).cast(pl.Float64).alias("longitude"),
        )
        if validate_flags:
            validate_station_quantity_flags(station_df, station)
        print_station_columns(
            station_name=station["FULL_NAME"],
            column_names=[
                column_name
                for column_name in station_df.columns
                if column_name not in STATION_DF_METADATA_COLUMNS
            ],
            name_width=station_name_width,
        )
        station_frames.append(station_df)

    if not station_frames:
        return pl.DataFrame()

    df = pl.concat(station_frames, how="diagonal_relaxed")
    return drop_all_null_columns(df=df, excluded_columns=STATION_DF_METADATA_COLUMNS)


def load_active_station_daily_data(
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    stations_data_dir=STATIONS_DATA_DAILY_PATH,
    start_date=None,
    end_date=None,
):
    """
    Load reshaped daily data for each active station.
    """
    return load_active_station_dataframe(
        station_loader=load_station_daily_dataframe,
        stations_csv_path=stations_csv_path,
        stations_data_dir=stations_data_dir,
        allow_missing=False,
        validate_flags=True,
        start_date=start_date,
        end_date=end_date,
    )


def load_active_station_hourly_data(
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    stations_data_dir=STATIONS_DATA_HOURLY_PATH,
    start_date=None,
    end_date=None,
):
    """
    Load reshaped hourly data for each active station.
    """
    return load_active_station_dataframe(
        station_loader=load_station_hourly_dataframe,
        stations_csv_path=stations_csv_path,
        stations_data_dir=stations_data_dir,
        allow_missing=True,
        validate_flags=False,
        start_date=start_date,
        end_date=end_date,
    )


def rename_hourly_overlap_columns(active_station_daily_df, active_station_hourly_df):
    """
    Rename hourly quantity columns that overlap with daily quantity columns by appending '1H'.
    """
    daily_columns = set(active_station_daily_df.columns) - STATION_DF_METADATA_COLUMNS
    overlapping_columns = [
        column_name
        for column_name in active_station_hourly_df.columns
        if column_name not in STATION_DF_METADATA_COLUMNS and column_name in daily_columns
    ]
    return active_station_hourly_df.rename(
        {column_name: f"{column_name}1H" for column_name in overlapping_columns}
    )


def load_cached_open_meteo_dataframe(
    latitude,
    longitude,
    site_id,
    start_date=None,
    end_date=None,
    data_dir=OPEN_METEO_DATA_HOURLY_PATH,
):
    """
    Read cached Open-Meteo JSON for the selected station location and shape it for storage update.
    """
    assert start_date is not None, "start_date must be provided for Open-Meteo cache loading."
    assert end_date is not None, "end_date must be provided for Open-Meteo cache loading."

    json_path = (
        Path(data_dir)
        / f"open-meteo-{site_id}-{start_date[:10]}-{end_date[:10]}.json"
    )
    open_meteo_df = open_meteo(
        latitude=latitude,
        longitude=longitude,
        start=start_date,
        end=end_date,
        json_path=json_path,
        refresh=False,
    )
    return open_meteo_df.with_columns(
        # pl.lit(f"open_meteo:{site_id}").alias("STATION"),
        pl.lit(latitude).cast(pl.Float64).alias("latitude"),
        pl.lit(longitude).cast(pl.Float64).alias("longitude"),
    ).select(
        [
            # "STATION",
            "date_time",
            "latitude",
            "longitude",
            *[
                column_name
                for column_name in open_meteo_df.columns
                if column_name != "date_time"
            ],
        ]
    )


def warn_long_pressure_gap_in_series(pressure_series, max_gap_hours=24):
    """
    Warn if a merged hourly pressure series still contains a null gap longer than max_gap_hours.
    """
    null_flags = pressure_series.isnull().values.tolist()
    longest_gap = 0
    current_gap = 0
    for is_null in null_flags:
        if is_null:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
        else:
            current_gap = 0

    if longest_gap > max_gap_hours:
        LOGGER.warning(
            "Merged hourly pressure series contains a null gap longer than %sh: %sh",
            max_gap_hours,
            longest_gap,
        )


def build_pressure_series(pressure_daily, pressure_hourly, max_gap_hours=24):
    """
    Build hourly pressure from preferred P1H and daily P interpolated onto the hourly time axis.
    """
    pressure_daily_valid = pressure_daily.dropna(dim="date_time")
    if pressure_daily_valid.sizes.get("date_time", 0) >= 2:
        pressure_daily_on_hourly = pressure_daily_valid.interp(
            date_time=pressure_hourly["date_time"],
            method="linear",
        )
    elif pressure_daily_valid.sizes.get("date_time", 0) == 1:
        pressure_daily_on_hourly = pressure_daily_valid.reindex(date_time=pressure_hourly["date_time"], method="nearest")
    else:
        pressure_daily_on_hourly = xr.full_like(pressure_hourly, np.nan)

    if pressure_daily_valid.sizes.get("date_time", 0) >= 1:
        max_gap = np.timedelta64(max_gap_hours, "h")
        time_axis = pressure_hourly["date_time"]
        first_time = pressure_daily_valid["date_time"].isel(date_time=0)
        last_time = pressure_daily_valid["date_time"].isel(date_time=-1)
        first_value = pressure_daily_valid.isel(date_time=0)
        last_value = pressure_daily_valid.isel(date_time=-1)

        early_tail_mask = (time_axis < first_time) & ((first_time - time_axis) <= max_gap)
        late_tail_mask = (time_axis > last_time) & ((time_axis - last_time) <= max_gap)

        pressure_daily_on_hourly = xr.where(early_tail_mask, first_value, pressure_daily_on_hourly)
        pressure_daily_on_hourly = xr.where(late_tail_mask, last_value, pressure_daily_on_hourly)

    pressure_series = pressure_hourly.fillna(pressure_daily_on_hourly)
    warn_long_pressure_gap_in_series(pressure_series, max_gap_hours=max_gap_hours)
    return pressure_series


def print_dataframe_column_diff(active_station_daily_df, active_station_hourly_df):
    """
    Print daily/hourly column overlap for visual inspection.
    """
    daily_columns = set(active_station_daily_df.columns) - STATION_DF_METADATA_COLUMNS
    hourly_columns = set(active_station_hourly_df.columns) - STATION_DF_METADATA_COLUMNS

    print("Common daily/hourly quantity columns:")
    print(sorted(daily_columns & hourly_columns))
    print("Daily-only quantity columns:")
    print(sorted(daily_columns - hourly_columns))
    print("Hourly-only quantity columns:")
    print(sorted(hourly_columns - daily_columns))


def update_storage(stations_df, node_path: list[str]):
    node, _ = read_storage(
        CHMI_STATIONS_SCHEMA_PATH,
        node_path=node_path,
        var_names=[],
        storage_path=CHMI_STATIONS_STORAGE_PATH,
    )
    node.update(stations_df)


def update_parflow_input_storage(
    parflow_clm_ds,
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    site_coords_csv_path=SITE_COORDS_CSV_PATH,
):
    """
    Save the ParFlow/CLM input dataset into the parflow_input zarr_fuse node.
    """
    site_metadata_ds = build_site_metadata_dataset(site_coords_csv_path=site_coords_csv_path)
    station_elevation_lookup = get_station_elevation_lookup(stations_csv_path=stations_csv_path)
    pressure_source_station = build_pressure_source_station_series(parflow_clm_ds)
    site_pressure = correct_pressure_to_site_elevations(
        pressure_pa=parflow_clm_ds["Press"],
        station_source=pressure_source_station,
        air_temperature_k=parflow_clm_ds["Temp"],
        site_elevation_m=site_metadata_ds["elevation"],
        station_elevation_lookup=station_elevation_lookup,
    ).transpose("date_time", "site_id").rename("Press")
    site_pressure.attrs = parflow_clm_ds["Press"].attrs

    ds_to_store = parflow_clm_ds.copy()
    ds_to_store["Press"] = site_pressure
    ds_to_store["latitude"] = site_metadata_asof(site_metadata_ds["latitude"], ds_to_store["date_time"])
    ds_to_store["longitude"] = site_metadata_asof(site_metadata_ds["longitude"], ds_to_store["date_time"])
    ds_to_store["elevation"] = site_metadata_asof(site_metadata_ds["elevation"], ds_to_store["date_time"])
    ds_to_store = ds_to_store.assign_coords(site_id=site_metadata_ds["site_id"])

    time_values = parflow_clm_ds["date_time"].values
    assert time_values.size >= 2, "Need at least two time steps to determine forcing interval."
    time_step = time_values[1] - time_values[0]
    time_interval = time_values[-1] - time_values[0]
    ds_to_store.attrs = {
        **ds_to_store.attrs,
        "time_step": time_step,
        "time_interval": time_interval,
    }


    node, _ = read_storage(
        CHMI_STATIONS_SCHEMA_PATH,
        node_path=NODE_PARFLOW,
        var_names=[],
        storage_path=CHMI_STATIONS_STORAGE_PATH,
    )
    node.update_from_ds(ds_to_store)


def get_priority_station_metadata(
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    station_priority=CLM_STATION_PRIORITY,
):
    """
    Resolve configured station names to metadata rows used for priority merging.
    """
    stations_df = pl.read_csv(stations_csv_path)
    priority_metadata = []
    for station_wsi in station_priority:
        station_rows = stations_df.filter(pl.col("WSI") == station_wsi)
        assert station_rows.height == 1, f"Expected exactly one station row for {station_wsi!r}."
        priority_metadata.append(station_rows.row(0, named=True))
    return priority_metadata


def get_station_elevation_lookup(stations_csv_path=ACTIVE_STATIONS_CSV_PATH):
    """
    Return a mapping from station WSI to station elevation in meters above sea level.
    """
    stations_df = pl.read_csv(stations_csv_path)
    return {
        row["WSI"]: float(row["ELEVATION"])
        for row in stations_df.select(["WSI", "ELEVATION"]).iter_rows(named=True)
    }


def build_pressure_source_station_series(parflow_clm_ds):
    """
    Reconstruct the station source series used for the merged pressure field.
    """
    hourly_source = parflow_clm_ds["P1H_source_station"]
    daily_source = parflow_clm_ds["P_source_station"]
    hourly_pressure = parflow_clm_ds["Press"]
    return xr.where(hourly_pressure.notnull() & hourly_source.notnull(), hourly_source, daily_source)


def build_site_metadata_dataset(site_coords_csv_path=SITE_COORDS_CSV_PATH):
    """
    Load site coordinates and elevations into an xarray dataset keyed by site_id and site_datetime.
    """
    site_metadata_df = load_site_coords_csv(str(site_coords_csv_path))
    site_ids = site_metadata_df.select(pl.col("site_id")).unique().sort("site_id")["site_id"].to_numpy()
    site_times = site_metadata_df.select("site_datetime").unique().sort("site_datetime")["site_datetime"].to_numpy()
    # site_ids = np.array(
    #     site_metadata_df.select(pl.col("site_id").cast(pl.Int32)).unique().sort("site_id")["site_id"].to_list(),
    #     dtype=np.int32,
    # )
    # site_times = np.array(
    #     site_metadata_df.select("site_datetime").unique().sort("site_datetime")["site_datetime"].to_list(),
    # )

    data_vars = {}
    for var_name in ["latitude", "longitude", "elevation"]:
        wide_df = (
            site_metadata_df
            .select(["site_datetime", "site_id", var_name])
            .pivot(
                values=var_name,
                index="site_datetime",
                on="site_id",
                aggregate_function="first",
                sort_columns=True,
            )
            .sort("site_datetime")
        )
        wide_df = wide_df.select(
            [
                "site_datetime",
                *[
                    pl.col(str(site_id)).cast(pl.Float64, strict=False)
                    for site_id in site_ids
                ],
            ]
        )
        data_vars[var_name] = (("site_datetime", "site_id"), wide_df.drop("site_datetime").to_numpy())

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "site_datetime": site_times,
            "site_id": site_ids,
        },
    )


def site_metadata_asof(site_metadata_da, date_time):
    """
    Resolve site metadata onto a target date_time axis using backward-looking "as of" selection.
    """
    if "site_datetime" not in site_metadata_da.dims:
        return site_metadata_da.expand_dims(date_time=date_time).transpose("date_time", "site_id")

    resolved_da = (
        site_metadata_da
        .rename(site_datetime="date_time")
        .reindex(date_time=date_time, method="ffill")
        .transpose("date_time", "site_id")
    )
    first_value = (
        site_metadata_da
        .isel(site_datetime=0)
        .drop_vars("site_datetime")
        .expand_dims(date_time=date_time)
        .transpose("date_time", "site_id")
    )
    return resolved_da.fillna(first_value)


def select_station_variable(ds, var_name, station):
    """
    Select one variable for one station and reduce it to a time series.
    """
    station_da = ds[var_name].sel(
        latitude=station["LAT"],
        longitude=station["LON"],
        method="nearest",
    )
    return station_da.squeeze(drop=True).drop_vars(
        [coord_name for coord_name in ["latitude", "longitude"] if coord_name in station_da.coords],
        errors="ignore",
    )


def merge_station_priority(ds, var_name, priority_metadata):
    """
    Merge one variable across stations using the configured station priority.
    """
    merged_da = None
    source_station = None
    for station in priority_metadata:
        station_da = select_station_variable(ds=ds, var_name=var_name, station=station)
        if merged_da is None:
            merged_da = station_da
            source_station = xr.DataArray(
                data=np.full(station_da.shape, station["WSI"], dtype=object),
                coords=station_da.coords,
                dims=station_da.dims,
                name=f"{var_name}_source_station",
            )
            continue

        fill_mask = merged_da.isnull() & station_da.notnull()
        merged_da = merged_da.combine_first(station_da)
        source_station = xr.where(fill_mask, station["WSI"], source_station)

    assert merged_da is not None, f"No station data available for variable {var_name!r}."
    station_names = [station["WSI"] for station in priority_metadata]
    merged_da.attrs["station_priority"] = station_names
    source_station.name = f"{var_name}_source_station"
    source_station.attrs["station_priority"] = station_names
    return merged_da, source_station


def compute_specific_humidity(temperature_c, relative_humidity_pct, pressure_hpa):
    """
    Convert temperature, relative humidity, and station pressure to specific humidity.
    Works with xarray DataArray inputs backed by chunked arrays.
    """
    saturation_vapor_pressure_hpa = 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))
    vapor_pressure_hpa = (relative_humidity_pct / 100.0) * saturation_vapor_pressure_hpa
    return 0.622 * vapor_pressure_hpa / (pressure_hpa - 0.378 * vapor_pressure_hpa)


def correct_pressure_to_site_elevations(
    pressure_pa,
    station_source,
    air_temperature_k,
    site_elevation_m,
    station_elevation_lookup,
):
    """
    Correct station pressure to all site elevations using the hypsometric relation.
    """
    station_elevation = xr.full_like(pressure_pa, np.nan, dtype=float)
    for station_wsi, elevation_m in station_elevation_lookup.items():
        station_elevation = xr.where(station_source == station_wsi, float(elevation_m), station_elevation)

    site_elevation = site_metadata_asof(site_elevation_m, pressure_pa["date_time"])
    height_delta = site_elevation - station_elevation

    return pressure_pa * np.exp(-(GRAVITY_M_S2 * height_delta) / (DRY_AIR_GAS_CONSTANT * air_temperature_k))


def with_clean_attrs(data_array, *, units, description, source_quantities):
    """
    Replace inherited attrs with a clean, explicit metadata set for CLM forcing variables.
    """
    data_array.attrs = {
        "units": units,
        "description": description,
        "source_quantities": source_quantities,
    }
    return data_array


def build_open_meteo_parflow_comparison_dataset(
    schema_path=CHMI_STATIONS_SCHEMA_PATH,
    storage_path=CHMI_STATIONS_STORAGE_PATH,
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
):
    """
    Read the stored Open-Meteo data and convert it to ParFlow/CLM-like quantities
    for comparison against parflow_clm_ds.
    """
    station = get_priority_station_metadata(stations_csv_path=stations_csv_path)[0]
    _, open_meteo_ds = read_storage(
        schema_path=schema_path,
        node_path=NODE_OPEN_METEO,
        var_names=OPEN_METEO_REQUIRED_VARS,
        storage_path=storage_path,
    )
    open_meteo_inputs = {
        var_name: select_station_variable(open_meteo_ds, var_name, station)
        for var_name in OPEN_METEO_REQUIRED_VARS
    }

    wind_direction_rad = np.deg2rad(open_meteo_inputs["wind_direction_10m"])
    pressure_hpa = open_meteo_inputs["surface_pressure"]
    template = open_meteo_inputs["temperature_2m"]
    return xr.Dataset(
        data_vars={
            "APCP": (open_meteo_inputs["precipitation"] / SECONDS_PER_HOUR).rename("APCP"),
            "Temp": (open_meteo_inputs["temperature_2m"] + 273.15).rename("Temp"),
            "UGRD": (-open_meteo_inputs["wind_speed_10m"] * np.sin(wind_direction_rad)).rename("UGRD"),
            "VGRD": (-open_meteo_inputs["wind_speed_10m"] * np.cos(wind_direction_rad)).rename("VGRD"),
            "Press": (pressure_hpa * 100.0).rename("Press"),
            "SPFH": compute_specific_humidity(
                open_meteo_inputs["temperature_2m"],
                open_meteo_inputs["relative_humidity_2m"],
                pressure_hpa,
            ).rename("SPFH"),
            "DSWR": open_meteo_inputs["shortwave_radiation"].rename("DSWR"),
            "DLWR": open_meteo_inputs["dlwr_estimate"].rename("DLWR"),
        },
        coords={"date_time": template["date_time"].values},
        attrs={"source": "Open-Meteo archive"},
    )


def plot_parflow_open_meteo_comparison(
    parflow_clm_ds,
    open_meteo_parflow_ds,
    output_path=PARFLOW_OPEN_METEO_COMPARISON_PLOT_PATH,
):
    """
    Create a quantity-by-quantity comparison plot between ParFlow/CLM forcing data
    and converted Open-Meteo data.
    """
    quantity_units = {
        "APCP": "mm/s",
        "Temp": "K",
        "UGRD": "m/s",
        "VGRD": "m/s",
        "Press": "Pa",
        "SPFH": "kg/kg",
        "DSWR": "W / m ** 2",
        "DLWR": "W / m ** 2",
    }

    # select only U01
    parflow_clm_ds = parflow_clm_ds.sel(site_id=1)

    figure, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 14), sharex=True)
    axes = axes.ravel()

    for axis, quantity_name in zip(axes, quantity_units, strict=False):
        parflow_da, open_meteo_da = xr.align(
            parflow_clm_ds[quantity_name],
            open_meteo_parflow_ds[quantity_name],
            join="inner",
        )
        parflow_mask = np.isfinite(parflow_da.values)
        open_meteo_mask = np.isfinite(open_meteo_da.values)
        axis.plot(
            parflow_da["date_time"].values[parflow_mask],
            parflow_da.values[parflow_mask],
            label="parflow_clm_ds",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
        )
        axis.plot(
            open_meteo_da["date_time"].values[open_meteo_mask],
            open_meteo_da.values[open_meteo_mask],
            label="open_meteo",
            linewidth=1.0,
            linestyle=":",
            alpha=0.75,
        )
        axis.set_title(quantity_name)
        axis.set_ylabel(quantity_units[quantity_name])
        axis.grid(True, alpha=0.3)
        if quantity_name == "DLWR":
            axis.text(
                0.5,
                0.5,
                "Open-Meteo DLWR unavailable",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        if quantity_name == "APCP":
            axis.legend(loc="upper right")

    figure.suptitle("ParFlow/CLM vs Open-Meteo comparison")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.show()
    # plt.close(figure)
    return output_path


def update_parflow_clm_from_open_meteo(
    parflow_clm_ds,
    open_meteo_parflow_ds,
    replace_quantities=("Temp", "UGRD", "VGRD", "SPFH", "DSWR", "DLWR"),
):
    """
    Replace selected ParFlow/CLM forcing quantities with aligned Open-Meteo quantities,
    while keeping APCP and Press from the CHMI-based dataset.
    """
    open_meteo_on_parflow_time = open_meteo_parflow_ds.reindex(date_time=parflow_clm_ds["date_time"])
    updated_ds = parflow_clm_ds.copy()

    for quantity_name in replace_quantities:
        updated_quantity = open_meteo_on_parflow_time[quantity_name].copy()
        if "site_id" in updated_ds[quantity_name].dims and "site_id" not in updated_quantity.dims:
            expanded_dims = [*updated_quantity.dims, "site_id"]
            updated_quantity = (
                updated_quantity
                .expand_dims(site_id=updated_ds["site_id"])
                .transpose(*expanded_dims)
            )
        updated_quantity.attrs = {
            **parflow_clm_ds[quantity_name].attrs,
            "replacement_source": "open_meteo",
        }
        updated_ds[quantity_name] = updated_quantity

    return updated_ds


def build_parflow_clm_input_dataset(
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
    schema_path=CHMI_STATIONS_SCHEMA_PATH,
    storage_path=CHMI_STATIONS_STORAGE_PATH,
    station_priority=CLM_STATION_PRIORITY,
):
    """
    Read stored CHMI station data, merge station priorities, and convert the result
    to a preliminary ParFlow/CLM forcing dataset.
    """
    _, source_ds = read_storage(
        schema_path=schema_path,
        node_path=NODE_CHMI_STATIONS,
        var_names=CLM_REQUIRED_SOURCE_VARS,
        storage_path=storage_path,
    )
    priority_metadata = get_priority_station_metadata(
        stations_csv_path=stations_csv_path,
        station_priority=station_priority,
    )

    merged_inputs = {}
    source_station_vars = {}
    for var_name in CLM_REQUIRED_SOURCE_VARS:
        merged_inputs[var_name], source_station_vars[var_name] = merge_station_priority(
            ds=source_ds,
            var_name=var_name,
            priority_metadata=priority_metadata,
        )

    merged_inputs = {
        var_name: data_array
        for var_name, data_array in merged_inputs.items()
    }
    source_station_vars = {
        var_name: data_array
        for var_name, data_array in source_station_vars.items()
    }

    wind_direction_rad = np.deg2rad(merged_inputs["D10"] * 10.0)
    template = merged_inputs["T"]
    pressure_series = build_pressure_series(
        pressure_daily=merged_inputs["P"],
        pressure_hourly=merged_inputs["P1H"],
    )
    parflow_clm_ds = xr.Dataset(
        data_vars={
            "APCP": with_clean_attrs(
                (merged_inputs["SRA1H"] / SECONDS_PER_DAY).rename("APCP"),
                units="mm/s",
                description="Total precipitation rate for ParFlow/CLM.",
                source_quantities=["SRA1H"],
            ),
            "Temp": with_clean_attrs(
                (merged_inputs["T"] + 273.15).rename("Temp"),
                units="K",
                description="Air temperature for ParFlow/CLM.",
                source_quantities=["T"],
            ),
            "UGRD": with_clean_attrs(
                (-merged_inputs["F"] * np.sin(wind_direction_rad)).rename("UGRD"),
                units="m/s",
                description="Eastward wind component for ParFlow/CLM.",
                source_quantities=["F", "D10"],
            ),
            "VGRD": with_clean_attrs(
                (-merged_inputs["F"] * np.cos(wind_direction_rad)).rename("VGRD"),
                units="m/s",
                description="Northward wind component for ParFlow/CLM.",
                source_quantities=["F", "D10"],
            ),
            "Press": with_clean_attrs(
                (pressure_series * 100.0).rename("Press"),
                units="Pa",
                description="Atmospheric pressure for ParFlow/CLM.",
                source_quantities=["P1H", "P"],
            ),
            "SPFH": with_clean_attrs(
                compute_specific_humidity(
                    merged_inputs["T"],
                    merged_inputs["H"],
                    pressure_series,
                ).rename("SPFH"),
                units="kg/kg",
                description="Specific humidity for ParFlow/CLM.",
                source_quantities=["T", "H", "P1H", "P"],
            ),
            "DSWR": with_clean_attrs(
                xr.full_like(template, np.nan).rename("DSWR"),
                units="W/m2",
                description="Downward shortwave radiation for ParFlow/CLM; unavailable from selected CHMI station variables.",
                source_quantities=[],
            ),
            "DLWR": with_clean_attrs(
                xr.full_like(template, np.nan).rename("DLWR"),
                units="W/m2",
                description="Downward longwave radiation for ParFlow/CLM; unavailable from selected CHMI station variables.",
                source_quantities=[],
            ),
        },
        coords={"date_time": template["date_time"].values},
        attrs={
            "station_priority": station_priority,
            "notes": (
                "Preliminary CHMI-to-ParFlow/CLM forcing conversion. "
                "DSWR and DLWR are unavailable from the selected station variables and remain NaN."
            ),
        },
    )

    for var_name, source_station in source_station_vars.items():
        parflow_clm_ds[source_station.name] = source_station

    site_ids = build_site_metadata_dataset()["site_id"]
    parflow_clm_ds = parflow_clm_ds.assign_coords(site_id=site_ids)

    for var_name, data_array in parflow_clm_ds.data_vars.items():
        if "date_time" not in data_array.dims or "site_id" in data_array.dims:
            continue
        expanded_dims = [*data_array.dims, "site_id"]
        parflow_clm_ds[var_name] = data_array.expand_dims(site_id=site_ids).transpose(*expanded_dims)

    return parflow_clm_ds


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    start_date = "2020-01-01T00:00:00Z"
    end_date = "2025-12-31T23:59:59Z"

    if not Path("chmi_stations.zarr").exists():
        active_station_daily_df = load_active_station_daily_data(start_date=start_date, end_date=end_date)
        # active_station_daily_df.sort(["date_time"])
        print("Active stations daily dataframe preview:")
        print(active_station_daily_df.head())
        print(f"Active stations daily dataframe shape: {active_station_daily_df.shape}")

        active_station_hourly_df = load_active_station_hourly_data(start_date=start_date, end_date=end_date)
        print_dataframe_column_diff(active_station_daily_df, active_station_hourly_df)
        print("Active stations hourly dataframe preview:")
        print(active_station_hourly_df.head())
        print(f"Active stations hourly dataframe shape: {active_station_hourly_df.shape}")

        active_station_hourly_df = rename_hourly_overlap_columns(active_station_daily_df,
                                                                 active_station_hourly_df)

        active_station_df = pl.concat(
            [active_station_daily_df, active_station_hourly_df],
            how="diagonal_relaxed",
        )
        update_storage(active_station_df, node_path=NODE_CHMI_STATIONS)

        # latitude, longitude, site_id = get_station_coordinates(active_station_df)
        latitude, longitude, site_id = (50.863565, 14.889853, '1')  # U01
        open_meteo_df = load_cached_open_meteo_dataframe(latitude, longitude, site_id,
                                                         start_date=start_date, end_date=end_date)
        print("Open-Meteo dataframe preview:")
        print(open_meteo_df.head())
        print(f"Open-Meteo dataframe shape: {open_meteo_df.shape}")
        update_storage(open_meteo_df, node_path=NODE_OPEN_METEO)

    # computes input fields for Parflow/CLM from chmi data
    parflow_clm_ds = build_parflow_clm_input_dataset()

    open_meteo_parflow_ds = build_open_meteo_parflow_comparison_dataset()
    comparison_plot_path = plot_parflow_open_meteo_comparison(
        parflow_clm_ds,
        open_meteo_parflow_ds,
    )

    # fill nulls from OpenMeteo source
    parflow_clm_ds = update_parflow_clm_from_open_meteo(
        parflow_clm_ds,
        open_meteo_parflow_ds,
    )
    update_parflow_input_storage(parflow_clm_ds)
    print(parflow_clm_ds)
    print(f"Saved ParFlow/Open-Meteo comparison plot to: {comparison_plot_path}")


if __name__ == "__main__":
    main()




# CHAT about snow measurements understanding:
#------------------------------------------------
# From the local CHMI metadata, the meanings are:
# •
# SRA: Srážka, unit mm (quantity_definitions.json)
# •
# SCE: Výška sněhu, unit cm
# •
# SCEdif: Denní rozdíl celkové výšky sněhu, unit cm
# •
# SNO: Nový sníh, unit cm
# •
# SVH: Vodní hodnota, unit mm
# That strongly implies:
# •
# SRA is precipitation amount, already in water equivalent.
# •
# SCE, SCEdif, SNO are snow depths, not precipitation totals.
# •
# SVH is snow water equivalent of the snowpack on the ground, a state variable, not incoming precipitation.
# So for CHMI station forcing, the simple model is:
# total_precipitation = SRA
# You should not compute total precipitation as SRA + snow columns, because that would mix flux with snowpack state/depth and double-count.
# Practical interpretation:
# •
# Use SRA as total precipitation input.
# •
# Use SNO as a snow diagnostic only.
# •
# Use SVH for snowpack storage / SWE checks.
# •
# Use SCE and SCEdif for snow depth diagnostics.
# One important correction: in your repo, chmi_stations_storage/chmi_stations/zarr.json currently describes SRA as Rainfall precipitation. That description looks misleading relative to the CHMI metadata, because Czech Srážka is precipitation generally, not rain-only.
# My recommendation:
# •
# treat SRA as total precipitation
# •
# do not derive total precipitation from SCE/SCEdif/SNO/SVH
# •
# if needed, split SRA into rain/snow later using temperature
# If you want, I can next patch the storage/schema description so SRA is labeled as total precipitation instead of rainfall.


#---------------------------------------------------
# For prepare_clm(), ParFlow/CLM wants exactly these meteorological forcings:
# •
# DSWR: downward shortwave radiation [W/m²]
# •
# DLWR: downward longwave radiation [W/m²]
# •
# APCP: precipitation rate [mm/s]
# •
# Temp: air temperature [K]
# •
# UGRD: eastward wind [m/s]
# •
# VGRD: northward wind [m/s]
# •
# Press: atmospheric pressure [Pa]
# •
# SPFH: specific humidity [kg/kg]
# Source: ParFlow docs, both the input keys and CLM setup pages:
# •
# https://parflow.readthedocs.io/en/latest/keys.html
# •
# https://parflow.readthedocs.io/en/latest/pfsystem.html
# Against your CHMI station variables, the practical mapping is:
# •
# APCP from SRA
# •
# Temp from T
# •
# UGRD, VGRD from F and D10
# •
# Press from P
# •
# SPFH from T, H, P
# •
# DSWR, DLWR are not directly available from the station set you listed
# •
# Snow columns SCE, SCEdif, SNO, SVH are diagnostics/state, not direct CLM met forcing inputs
# Recommended transforms:
# •
# APCP = SRA / Δt
# ◦
# If SRA is a daily total in mm, use Δt = 86400 s
# ◦
# This is acceptable for daily forcing, but poor for subdaily CLM because it destroys storm timing
# •
# Temp = T + 273.15
# •
# Press = P * 100
# ◦
# P is station pressure in hPa; this is better than sea-level pressure for CLM
# •
# Wind from meteorological direction:
# ◦
# theta = D10 * 10 * π / 180
# ◦
# UGRD = -F * sin(theta)
# ◦
# VGRD = -F * cos(theta)
# ◦
# This is an inference from standard meteorological convention
# •
# Specific humidity from RH:
# ◦
# es(T) = 6.112 * exp(17.67*T / (T + 243.5)) in hPa
# ◦
# e = (H/100) * es
# ◦
# q = 0.622 * e / (P - 0.378*e) in kg/kg
# ◦
# Here T is in °C, P in hPa
# ◦
# This conversion is standard meteorology, not a ParFlow-specific formula
# What to do with the snow variables:
# •
# Do not feed SCE, SCEdif, SNO, or SVH as CLM forcing columns
# •
# CLM internally partitions precipitation into rain/snow from met forcing; ParFlow docs explicitly expose snow partition options like Solver.CLM.SnowPartition, SnowTLow, SnowTHigh, wet-bulb methods, etc.
# •
# A useful use of SVH is as an initial snow-water-equivalent state if you want to initialize snowpack consistently. This is an inference from CLM practice and from your commented h2osno_ini in parflow_model.py
# What is missing from your station data:
# •
# DSWR: no direct shortwave radiation variable in your active list
# •
# DLWR: no direct longwave radiation variable in your active list
# Possible approximations, but I would treat them as fallback only:
# •
# DSWR from SSV using Angstrom-Prescott daily solar radiation estimation
# •
# DLWR from T, vapor pressure, and a cloudiness proxy using an empirical clear-sky/cloud correction
# Those are common meteorological approximations, but they are not ParFlow/CLM-native requirements and would introduce substantial structural uncertainty, especially for daily station data.
# My recommendation:
# 1.
# Use station data directly for APCP, Temp, Press, UGRD, VGRD, SPFH.
# 2.
# Do not use snow columns as forcing; use them only for validation or initial snow state.
# 3.
# Do not force CLM from these stations alone unless you also provide defensible DSWR and DLWR.
# 4.
# Best practical path: use your gridded atmospheric product for radiation and maybe humidity/wind, and use CHMI stations to bias-correct precipitation, temperature, and pressure.
# One important code issue in your current prepare_clm() if you adapt it to station data:
# •
# VGRD is currently computed with cos again; it should use sin/cos as above
# •
# SPFH cannot be set equal to relative humidity
# •
# Press should use station pressure if you have it, not sea-level pressure
# •
# The final file write is iterating over ds.data_vars.values() instead of clm_met_data
# If you want, I can turn this into a concrete mapping spec for your CHMI zarr variables and patch prepare_clm() accordingly.
