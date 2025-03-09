"""
Scraping data from yr.no for Uhelna, Czech Republic
"""
from datetime import datetime

import numpy as np
import polars as pl
from pathlib import Path
import requests
import requests_cache


from zarr_storage import create, update, read

def grid_points():
    min = (50.840000, 14.850000)
    max = (50.8900000, 14.9600000)

    step = 1e-2 # about 1000m; approximately resolution of the yr.no service
    lon_rng = np.arange(min[0], max[0], step)
    lat_rng = np.arange(min[1], max[1], step)
    lon, lat =  np.meshgrid(lon_rng, lat_rng)
    return lon.flatten(), lat.flatten()

def sensor_locations():
    path = Path() / 'data' / 'seznam_stanovist.csv'
    raw_df = pl.read_csv(path, has_header=True)
    lonlat = raw_df['souřadnice']
    lon, lat = zip(*[row.split(',') for row in lonlat if row is not None])
    lon = np.array(lon, dtype=float)
    lat = np.array(lat, dtype=float)
    return lon, lat

def location_df():
    grid_lon, grid_lat = grid_points()
    grid_flag = np.ones_like(grid_lon, dtype=bool)
    sen_lon, sen_lat = sensor_locations()
    grid_flag_sen = np.zeros_like(sen_lon, dtype=bool)
    lon = np.concatenate([grid_lon, sen_lon])
    lat = np.concatenate([grid_lat, sen_lat])
    grid_flag = np.concatenate([grid_flag, grid_flag_sen])
    df = pl.DataFrame({'lon': lon, 'lat': lat, 'grid': grid_flag})
    return df

def create_zarr(zarr_path: Path):
    grid_lon, grid_lat = grid_points()
    grid_flag = np.ones_like(grid_lon, dtype=bool)
    sen_lon, sen_lat = sensor_locations()
    grid_flag_sen = np.zeros_like(sen_lon, dtype=bool)
    lon = np.concatenate([grid_lon, sen_lon])
    lat = np.concatenate([grid_lat, sen_lat])
    grid_flag = np.concatenate([grid_flag, grid_flag_sen])
    loc_struct = pl.DataFrame(dict(lon=lon, lat=lat)).to_struct()
    df = pl.DataFrame({
        "time": datetime.now(),
        'loc': loc_struct,
        'grid': grid_flag})
    indices = {
        'time': 0,
        'loc': 200,
    }
    # TODO: grid as non-dim coordinate
    create(zarr_path, df, indices)


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

    cache = requests_cache.CachedSession('http_cache', expire_after=None)  # No expiration (permanent cache)
    response = cache.get(url, headers=headers, params=params)
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

def loc_forecast(lon, lat, grid):
    df = get_3day_forecast_yr(lon, lat)
    df = df.with_columns(
        location=pl.struct(lon=lon, lat=lat, grid=grid)
    )
    print(df)
    return df

def update_from_yr_no(zarr_path: Path):
    loc_df = location_df()
    if not zarr_path.exists():
        create_zarr(zarr_path)

    loc_dfs = [loc_forecast(** loc) for loc in loc_df.iter_rows(named=True)]
    # Concatenate all DataFrames
    final_df = pl.concat(loc_dfs)

    df = final_df

    # Total number of unique locations
    total_locs = df["location"].n_unique()
    # Count unique 'loc' per 'time'
    time_counts = df.group_by("date_time").agg(pl.col("location").n_unique().alias("loc_count"))
    # Find first time where all locations are present
    first_complete_time = time_counts.filter(pl.col("loc_count") == total_locs)["date_time"].min()

    # Filter only the first time and select 'air_temperature' for all 'loc' values
    result = df.filter(pl.col("date_time") == first_complete_time)

    # Print result
    print(result)

    update(zarr_path, final_df)

def main():

    zarr_path = Path() / 'data' / 'grid_sensors.zarr'
    update_from_yr_no(zarr_path)

if __name__ == '__main__':
    main()