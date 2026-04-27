import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import pandas as pd

from hlavo.ingress.meteo_playground.chmi_stations.config import (
    ACTIVE_STATIONS_CSV_PATH,
    CLM_STATION_PRIORITY,
    HISTORICAL_DAILY_URL,
    HISTORICAL_HOURLY_URL,
    OPEN_METEO_DATA_HOURLY_PATH,
    STATIONS_DATA_DAILY_PATH,
    STATIONS_DATA_HOURLY_PATH,
)
from hlavo.ingress.meteo_playground.chmi_stations.open_meteo import open_meteo

LOGGER = logging.getLogger(__name__)


def fetch_directory_filenames(index_url):
    """
    Load a directory listing and return linked filenames.
    """
    with urlopen(index_url) as response:
        html = response.read().decode("utf-8")

    filenames = re.findall(r'href="([^"]+)"', html)
    return [
        filename
        for filename in filenames
        if filename not in {"../", "./"} and not filename.endswith("/")
    ]


def find_station_data_urls(active_df, index_url):
    """
    Match CHMI data filenames whose names contain active station WSI codes.
    """
    if active_df.empty:
        return []

    active_wsis = active_df["WSI"].dropna().astype(str).unique().tolist()
    filenames = fetch_directory_filenames(index_url)
    matching_filenames = sorted(
        filename
        for filename in filenames
        if any(wsi in filename for wsi in active_wsis)
    )
    return [urljoin(index_url, filename) for filename in matching_filenames]


def download_station_data(file_urls, output_dir):
    """
    Download matched CHMI files into a local directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []
    total_files = len(file_urls)
    for index, file_url in enumerate(file_urls, start=1):
        print(f"\r{index}/{total_files} files downloading", end="", flush=True)
        target_path = output_path / Path(file_url).name
        with urlopen(file_url) as response:
            target_path.write_bytes(response.read())
        downloaded_paths.append(target_path)
        # LOGGER.info("Downloaded %s", target_path.name)
    if total_files > 0:
        print()

    return downloaded_paths


def load_active_stations(stations_csv_path=ACTIVE_STATIONS_CSV_PATH):
    """
    Load the already prepared active-stations CSV from meta_processing.py.
    """
    return pd.read_csv(stations_csv_path)


def get_station_coordinates(active_df, station_wsi=None):
    """
    Resolve latitude and longitude for the selected active station.
    """
    if station_wsi is None:
        station_wsi = CLM_STATION_PRIORITY[0]

    station_rows = active_df.loc[active_df["WSI"].astype(str) == str(station_wsi)]
    assert len(station_rows) == 1, f"Expected exactly one active station row for {station_wsi!r}."
    station = station_rows.iloc[0]
    return float(station["LAT"]), float(station["LON"]), str(station_wsi)


def download_chmi_station_data(
    active_df,
    *,
    daily_output_dir=STATIONS_DATA_DAILY_PATH,
    hourly_output_dir=STATIONS_DATA_HOURLY_PATH,
    hourly_years=("2024", "2025"),
):
    """
    Download CHMI daily and hourly files for the active stations.
    """
    LOGGER.info("Downloading CHMI daily data ...")
    daily_urls = find_station_data_urls(active_df, HISTORICAL_DAILY_URL)
    downloaded_daily = download_station_data(daily_urls, daily_output_dir)

    downloaded_hourly = []
    for year in hourly_years:
        LOGGER.info(f"Downloading CHMI hourly data for year {year} ...")
        hourly_urls = find_station_data_urls(active_df, f"{HISTORICAL_HOURLY_URL}{year}/")
        downloaded_hourly.extend(download_station_data(hourly_urls, Path(hourly_output_dir) / year))

    return downloaded_daily, downloaded_hourly


def download_open_meteo_data(
    *,
    latitude,
    longitude,
    start_date,
    end_date,
    site_id="site",
    output_dir=OPEN_METEO_DATA_HOURLY_PATH,
    refresh=False,
):
    """
    Download or refresh cached Open-Meteo JSON for the selected coordinates.
    """
    json_path = (
        Path(output_dir)
        / f"open-meteo-{site_id}-{start_date[:10]}-{end_date[:10]}.json"
    )
    open_meteo(
        latitude=latitude,
        longitude=longitude,
        start=start_date,
        end=end_date,
        json_path=json_path,
        refresh=refresh,
    )
    return json_path


def main(start_date: str, end_date: str):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_year = datetime.fromisoformat(start_date).year
    end_year = datetime.fromisoformat(end_date).year
    hourly_years = [str(year) for year in range(start_year, end_year + 1)]

    active_df = load_active_stations()

    downloaded_daily, downloaded_hourly = download_chmi_station_data(
        active_df,
        hourly_years=hourly_years,
    )
    print(f"Downloaded {len(downloaded_daily)} daily CHMI files to {STATIONS_DATA_DAILY_PATH}")
    print(f"Downloaded {len(downloaded_hourly)} hourly CHMI files to {STATIONS_DATA_HOURLY_PATH}")

    # latitude, longitude, site_id = get_station_coordinates(active_df) # top priority station
    latitude, longitude, site_id = (50.863565, 14.889853, '1') # U01
    open_meteo_json_path = download_open_meteo_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        site_id=site_id,
    )
    print(f"Cached Open-Meteo JSON to {open_meteo_json_path}")
