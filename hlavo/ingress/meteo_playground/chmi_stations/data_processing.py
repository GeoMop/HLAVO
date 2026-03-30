import json
import logging
from pathlib import Path

import polars as pl

from hlavo.common.zarr_fuse_reader import read_storage

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
    station_df_metadata_columns = {"STATION", "VTYPE", "date_time", "latitude", "longitude"}

    quantity_columns = [
        column_name
        for column_name in station_df.columns
        if column_name not in station_df_metadata_columns
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


def load_active_station_data(
    stations_csv_path="stations_nearby_active.csv",
    stations_data_dir="stations_data",
):
    """
    Load reshaped daily data for each active station, append station coordinates,
    concatenate the result, and drop columns that are entirely null.
    """
    stations_df = pl.read_csv(stations_csv_path)

    station_frames = []
    for station in stations_df.iter_rows(named=True):
        station_df = load_station_daily_dataframe(
            wsi=station["WSI"],
            stations_data_dir=stations_data_dir,
        )
        print(station["FULL_NAME"], station_df.columns[3:])
        validation = validate_station_quantity_flags(station_df, station)

        station_df = station_df.with_columns(
            pl.lit(station["LAT"]).cast(pl.Float64).alias("latitude"),
            pl.lit(station["LON"]).cast(pl.Float64).alias("longitude"),
        )
        station_frames.append(station_df)

    if not station_frames:
        return pl.DataFrame()

    df = pl.concat(station_frames, how="diagonal_relaxed")
    non_empty_columns = [
        column_name
        for column_name in df.columns
        if df.select(pl.col(column_name).is_not_null().any()).item()
    ]
    return df.select(non_empty_columns)


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
