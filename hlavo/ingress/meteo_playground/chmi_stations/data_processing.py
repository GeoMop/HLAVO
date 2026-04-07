import json
import logging
from pathlib import Path

import polars as pl
import xarray as xr
import numpy as np

from hlavo.common.zarr_fuse_reader import read_storage

LOGGER = logging.getLogger(__name__)

STATION_DF_METADATA_COLUMNS = {"STATION", "VTYPE", "date_time", "latitude", "longitude"}
SCRIPT_DIR = Path(__file__).resolve().parent
CHMI_STATIONS_SCHEMA_PATH = SCRIPT_DIR / "chmi_stations_schema.yaml"
CHMI_STATIONS_STORAGE_PATH = SCRIPT_DIR / "chmi_stations_storage"
ACTIVE_STATIONS_CSV_PATH = SCRIPT_DIR / "stations_nearby_active.csv"
STATIONS_DATA_DAILY_PATH = SCRIPT_DIR / "stations_data_daily"
STATIONS_DATA_HOURLY_PATH = SCRIPT_DIR / "stations_data_hourly"
CLM_STATION_PRIORITY = ["0-203-0-20407036001", #"Chotyně"
                        "0-203-0-11601", # "Frýdlant"
                        "0-20000-0-11603", # "Liberec"
                        ]
CLM_REQUIRED_SOURCE_VARS = ["SRA1H", "T", "P", "P1H", "F", "D10", "H"]
SECONDS_PER_DAY = 24 * 60 * 60


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
    pressure_daily_valid = pressure_daily.dropna(dim="time")
    if pressure_daily_valid.sizes.get("time", 0) >= 2:
        pressure_daily_on_hourly = pressure_daily_valid.interp(
            time=pressure_hourly["time"],
            method="linear",
        )
    elif pressure_daily_valid.sizes.get("time", 0) == 1:
        pressure_daily_on_hourly = pressure_daily_valid.reindex(time=pressure_hourly["time"], method="nearest")
    else:
        pressure_daily_on_hourly = xr.full_like(pressure_hourly, np.nan)

    if pressure_daily_valid.sizes.get("time", 0) >= 1:
        max_gap = np.timedelta64(max_gap_hours, "h")
        time_axis = pressure_hourly["time"]
        first_time = pressure_daily_valid["time"].isel(time=0)
        last_time = pressure_daily_valid["time"].isel(time=-1)
        first_value = pressure_daily_valid.isel(time=0)
        last_value = pressure_daily_valid.isel(time=-1)

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


def update_storage(stations_df):
    node, _ = read_storage(
        CHMI_STATIONS_SCHEMA_PATH,
        node_path=["chmi_stations"],
        var_names=[],
        storage_path=CHMI_STATIONS_STORAGE_PATH,
    )

    # node.update(add_missing_schema_columns(stations_df, node))
    node.update(stations_df)


def update_parflow_input_storage(
    parflow_clm_ds,
    stations_csv_path=ACTIVE_STATIONS_CSV_PATH,
):
    """
    Save the ParFlow/CLM input dataset into the parflow_input zarr_fuse node.
    """
    node, _ = read_storage(
        CHMI_STATIONS_SCHEMA_PATH,
        node_path=["parflow_input"],
        var_names=[],
        storage_path=CHMI_STATIONS_STORAGE_PATH,
    )
    priority_metadata = get_priority_station_metadata(stations_csv_path=stations_csv_path)
    anchor_station = priority_metadata[0]

    ds_to_store = parflow_clm_ds.rename({"time": "date_time"}).assign_coords(
        date_time=("date_time", parflow_clm_ds["time"].values),
        latitude=("latitude", [anchor_station["LAT"]]),
        longitude=("longitude", [anchor_station["LON"]]),
    )
    node.update_from_ds(ds_to_store)


def read_station_storage(
    schema_path=CHMI_STATIONS_SCHEMA_PATH,
    storage_path=CHMI_STATIONS_STORAGE_PATH,
    var_names=None,
):
    """
    Read station data from zarr_fuse storage.
    """
    read_var_names = [] if var_names is None else var_names
    _, ds = read_storage(
        schema_path=schema_path,
        node_path=["chmi_stations"],
        var_names=read_var_names,
        storage_path=storage_path,
    )
    return ds


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
    source_ds = read_station_storage(
        schema_path=schema_path,
        storage_path=storage_path,
        var_names=CLM_REQUIRED_SOURCE_VARS,
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
        var_name: data_array.rename({"date_time": "time"})
        for var_name, data_array in merged_inputs.items()
    }
    source_station_vars = {
        var_name: data_array.rename({"date_time": "time"})
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
        coords={"time": template["time"].values},
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

    time_values = parflow_clm_ds["time"].values
    assert time_values.size >= 2, "Need at least two time steps to determine forcing interval."
    parflow_clm_ds["time_step"] = xr.DataArray(time_values[1] - time_values[0])
    parflow_clm_ds["time_interval"] = xr.DataArray(time_values[-1] - time_values[0])

    return parflow_clm_ds


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    start_date = "2024-01-01T00:00:00Z"
    end_date = "2025-12-31T23:59:59Z"

    if not Path("chmi_stations_storage").exists():
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

        active_station_df = pl.concat([active_station_daily_df, active_station_hourly_df], how="diagonal_relaxed")
        update_storage(active_station_df)

    parflow_clm_ds = build_parflow_clm_input_dataset()
    update_parflow_input_storage(parflow_clm_ds)
    print(parflow_clm_ds)

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
