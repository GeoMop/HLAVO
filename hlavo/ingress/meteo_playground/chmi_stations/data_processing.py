import json
import logging
from pathlib import Path

import polars as pl

from hlavo.common.zarr_fuse_reader import read_storage

LOGGER = logging.getLogger(__name__)

STATION_DF_METADATA_COLUMNS = {"STATION", "VTYPE", "date_time", "latitude", "longitude"}


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


def load_station_daily_dataframe(wsi, stations_data_dir="stations_data"):
    """
    Read one downloaded CHMI daily data file, pivot ELEMENT values into columns,
    sort by datetime, and keep only years 2024 and 2025.
    """
    station_path = Path(stations_data_dir) / f"dly-{wsi}.json"
    if not station_path.exists():
        raise FileNotFoundError(f"Station data file not found: {station_path}")

    with station_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    data_block = get_data_block(doc)
    header = data_block["header"].split(",")
    values = data_block["values"]

    df = pl.DataFrame(values, schema=header, orient="row")
    df = df.with_columns(
        pl.col("DT").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False),
        pl.col("VAL").cast(pl.Utf8).str.strip_chars().replace("", None).cast(pl.Float64, strict=False),
    )

    df = df.pivot(
        on="ELEMENT",
        values="VAL",
        index=["STATION", "VTYPE", "DT"],
        aggregate_function="first",
    )

    return df.filter(
        pl.col("DT").dt.year().is_between(2020, 2025, closed="both")
    ).sort("DT").rename({"DT": "date_time"})


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


def load_active_station_data(
    stations_csv_path="stations_nearby_active.csv",
    stations_data_dir="stations_data",
):
    """
    Load reshaped daily data for each active station, append station coordinates,
    concatenate the result, and drop columns that are entirely null.
    """
    stations_df = pl.read_csv(stations_csv_path)
    station_name_width = stations_df.select(pl.col("FULL_NAME").str.len_chars().max()).item()

    station_frames = []
    for station in stations_df.iter_rows(named=True):
        station_df = load_station_daily_dataframe(
            wsi=station["WSI"],
            stations_data_dir=stations_data_dir,
        )
        station_df = station_df.with_columns(
            pl.lit(station["LAT"]).cast(pl.Float64).alias("latitude"),
            pl.lit(station["LON"]).cast(pl.Float64).alias("longitude"),
        )
        validate_station_quantity_flags(station_df, station)
        # station_df = drop_all_null_columns(df=station_df, excluded_columns=STATION_DF_METADATA_COLUMNS)
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


def update_storage(stations_df):
    node, ds = read_storage("chmi_stations_schema.yaml",
                            node_path=["chmi_stations"], var_names=[], storage_path="chmi_stations_storage")

    node.update(stations_df)
    pass


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    active_station_df = load_active_station_data()
    print("Active stations daily dataframe preview:")
    print(active_station_df.head())
    print(f"Active stations daily dataframe shape: {active_station_df.shape}")

    update_storage(active_station_df)

    pass


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