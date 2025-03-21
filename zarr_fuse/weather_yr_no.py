from typing import List, Tuple
import requests
from pathlib import Path
import polars as pl
from datetime import datetime, timezone


def get_3day_forecast_yr(lat: float, lon: float) -> pl.DataFrame:
    """
    Fetch the complete forecast data from yr.no and flatten it into a Polars DataFrame.

    This function queries the 'complete' endpoint from the Norwegian Meteorological Institute.
    It processes the returned JSON to build a DataFrame with:
      - 'date_time': forecast timestamp (as a datetime object).
      - All key-value pairs from the "instant" -> "details" section.
      - All key-value pairs from the "next_1_hours" -> "details" section, with keys prefixed by "n1h_".

    Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        pl.DataFrame: A flattened DataFrame containing the forecast data.
    """
    url = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
    headers = {
        "User-Agent": "MyWeatherApp/1.0 (your_email@example.com)"  # Replace with your details.
    }
    params = {
        "lat": lat,
        "lon": lon
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    timeseries = data.get("properties", {}).get("timeseries", [])

    records = []
    for entry in timeseries:
        # Parse the forecast time. Adjust "Z" to an ISO-compatible offset.
        time_str = entry.get("time")
        # Convert the time string to a datetime object.
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        record = {"date_time": dt}

        # Flatten the "instant" details.
        instant_details = entry.get("data", {}).get("instant", {}).get("details", {})
        record.update(instant_details)

        # Flatten the "next_1_hours" details, if available, with a "n1h_" prefix.
        n1h_details = entry.get("data", {}).get("next_1_hours", {}).get("details", {})
        for key, value in n1h_details.items():
            record[f"{key}"] = value

        records.append(record)

    # Create a Polars DataFrame from the list of records.
    df = pl.DataFrame(records)
    return df

#
# def last_current_merged(work_dir, locations):
#     """
#     Read the latest forecast data for the specified locations and merge it with the current weather data.
#
#     This function reads the latest forecast data for each location from the 'yr.no' API.
#     It then reads the current weather data for each location from the 'meteostat' API.
#     The two datasets are merged based on the location coordinates.
#
#     Parameters:
#         work_dir (str): Path to the working directory.
#         locations (list): List of location dictionaries, each containing:
#             - 'name': Name of the location.
#             - 'lat': Latitude of the location.
#             - 'lon': Longitude of the location.
#
#     Returns:
#         pl.DataFrame: A merged DataFrame containing the latest forecast and current weather data for each location.
#     """
#
# last_raw_forcast_file = "last_yr_forcast.csv"
# def update_yr_forecasts(locations: List[Tuple[float, float]], work_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
#     """
#     Update the historical forecast dataset by scraping the current 3-day forecasts
#     for a list of locations and merging them with the previously saved forecasts.
#
#     This function performs the following steps:
#       1. For each (lat, lon) in `locations`, fetch the 3-day forecast using the
#          get_3day_forecast_yr_polars function (assumed to be defined elsewhere),
#          add columns "lat" and "lon", and merge all results into a single Polars DataFrame
#          (raw_current).
#       2. Load saved forecasts from 'last_yr_forcast.csv' (if it exists in work_dir);
#          otherwise, use an empty DataFrame.
#       3. Concatenate the saved forecasts with the raw current forecasts and filter out
#          any rows with forecast times in the future (i.e. keep only rows with date_time <= current UTC time).
#       4. Return both the raw current DataFrame and the merged historical DataFrame.
#
#     Parameters:
#         locations (List[Tuple[float, float]]): List of (lat, lon) pairs.
#         work_dir (Path): A Path object pointing to the directory where 'last_yr_forcast.csv' is stored.
#
#     Returns:
#         Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
#           - The raw current forecast DataFrame.
#           - The merged historical forecast DataFrame.
#     """
#     # 1. Get raw current forecasts for all locations.
#     raw_dfs: List[pl.DataFrame] = []
#     for lat, lon in locations:
#         # Assume get_3day_forecast_yr_polars is defined elsewhere.
#         df_loc = get_3day_forecast_yr(lat, lon)
#         # Add lat and lon columns.
#         df_loc = df_loc.with_columns([
#             pl.lit(lat).alias("lat"),
#             pl.lit(lon).alias("lon")
#         ])
#         raw_dfs.append(df_loc)
#
#     raw_current: pl.DataFrame = pl.concat(raw_dfs) if raw_dfs else pl.DataFrame()
#
#     # 2. Load saved forecasts from CSV, if available.
#     last_file: Path = work_dir / last_raw_forcast_file
#     if last_file.exists():
#         try:
#             last_df: pl.DataFrame = pl.read_csv(last_file, parse_dates=True)
#         except Exception as e:
#             print("Error reading last forecast file:", e)
#             last_df = pl.DataFrame()
#     else:
#         last_df = pl.DataFrame()
#
#     # 3. Merge the saved forecasts with the current forecast.
#     union_df: pl.DataFrame = pl.concat([last_df, raw_current]) if last_df.height > 0 else raw_current
#
#     # Use a timezone-aware current UTC time.
#     now_utc: datetime = datetime.now(timezone.utc)
#     # Filter out future times by ensuring that both sides of the comparison are timezone-aware.
#     merged_df: pl.DataFrame = union_df.filter(pl.col("date_time") <= pl.lit(now_utc))
#
#     return raw_current, merged_df
#
#
# def update_and_save_yr_history(locations: List[Tuple[float, float]], work_dir: Path) -> Tuple[
#     pl.DataFrame, pl.DataFrame]:
#     """
#     Update and save the historical yr.no forecast dataset.
#
#     This function performs the following steps:
#
#       1. Calls update_yr_forecasts(locations, work_dir) to obtain:
#            - raw_current: the raw current 3-day forecasts (a Polars DataFrame),
#            - merged_df: a merged DataFrame containing all historical forecast rows (with forecast times <= current UTC).
#       2. Loads existing historical data from the file 'yr_no_last_history.csv' in work_dir (if it exists);
#          otherwise, it creates an empty Polars DataFrame.
#       3. Updates the historical data by concatenating the loaded history with merged_df and then ensuring that
#          for each unique key (date_time, lon, lat) only the latest row (from merged_df) is kept.
#          The result is called history_df.
#       4. For each complete day in history_df (i.e. any day strictly before today's UTC date),
#          writes that day's data to a separate file named 'yr_no_history_{date}.csv' and removes those rows
#          from the overall history.
#       5. Writes the remaining (incomplete) historical data to 'yr_no_last_history.csv'.
#       6. Saves raw_current to 'last_yr_forcast.csv' (to be loaded by update_yr_forecasts on the next run).
#
#     Parameters:
#         locations (List[Tuple[float, float]]): List of (lat, lon) pairs.
#         work_dir (Path): Directory where history files will be read/written.
#
#     Returns:
#         Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - raw_current: The raw current forecast DataFrame.
#             - merged_df: The merged historical forecast DataFrame (only data up to current UTC).
#
#     Note:
#         This function assumes that a function `update_yr_forecasts(locations, work_dir)` is defined elsewhere,
#         which returns (raw_current, merged_df) as Polars DataFrames.
#     """
#     # Step 1: Get current forecasts.
#     raw_current, merged_df = update_yr_forecasts(locations, work_dir)
#
#     # Step 2: Load existing historical data from 'yr_no_last_history.csv'
#     history_file: Path = work_dir / "yr_no_last_history.csv"
#     if history_file.exists():
#         try:
#             hist_df: pl.DataFrame = pl.read_csv(history_file, parse_dates=True)
#         except Exception as e:
#             print("Error reading historical file:", e)
#             hist_df = pl.DataFrame()
#     else:
#         hist_df = pl.DataFrame()
#
#     # Step 3: Merge existing historical data with merged_df.
#     # To ensure that merged_df overrides any duplicates, we concatenate
#     # [hist_df, merged_df] and then keep the last occurrence per (date_time, lon, lat).
#     union_df: pl.DataFrame = pl.concat([hist_df, merged_df])
#     # Reverse the DataFrame so that later rows come first, then take unique based on keys,
#     # then reverse back to restore order.
#     history_df: pl.DataFrame = union_df.reverse().unique(
#         subset=["date_time", "lon", "lat"], maintain_order=True
#     ).reverse()
#
#     # Step 4: For complete days (any day strictly before today's UTC date), write each to a separate file.
#     now_utc: datetime = datetime.now(timezone.utc)
#     today_date = now_utc.date()
#     # Add a temporary column "date_str" based on the "date_time" column (formatted as YYYY-MM-DD).
#     history_df = history_df.with_columns(
#         pl.col("date_time").dt.strftime("%Y-%m-%d").alias("date_str")
#     )
#
#     # Get the unique day strings in the historical data.
#     unique_dates = history_df.select("date_str").unique().to_series().to_list()
#     # Identify complete days (those strictly before today's date).
#     complete_days = [d for d in unique_dates if d < today_date.isoformat()]
#
#     # Filter out rows for complete days to form the remaining (incomplete) history.
#     remaining_rows = history_df.filter(~pl.col("date_str").is_in(complete_days))
#
#     # For each complete day, write its data to a separate CSV file.
#     for d in complete_days:
#         day_df = history_df.filter(pl.col("date_str") == d)
#         file_name = work_dir / f"yr_no_history_{d}.csv"
#         day_df.drop("date_str").write_csv(file_name)
#
#     # Step 5: Write the remaining (incomplete) history data to 'yr_no_last_history.csv'
#     remaining_rows.drop("date_str").write_csv(history_file)
#
#     # Step 6: Save raw_current to 'last_yr_forcast.csv' for next update.
#     raw_file: Path = work_dir / last_raw_forcast_file
#     raw_current.write_csv(raw_file)
#
#     return raw_current, merged_df
#
