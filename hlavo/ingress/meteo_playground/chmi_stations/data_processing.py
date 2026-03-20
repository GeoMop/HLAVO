import json
from pathlib import Path

import polars as pl


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
        pl.col("DT").dt.year().is_between(2024, 2025, closed="both")
    ).sort("DT").rename({"DT": "date_time"})


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
    for station in stations_df.select(["WSI", "LAT", "LON"]).iter_rows(named=True):
        station_df = load_station_daily_dataframe(
            wsi=station["WSI"],
            stations_data_dir=stations_data_dir,
        ).with_columns(
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


def main():
    active_station_df = load_active_station_data()
    print("Active stations daily dataframe preview:")
    print(active_station_df.head())
    print(f"Active stations daily dataframe shape: {active_station_df.shape}")


if __name__ == "__main__":
    main()
