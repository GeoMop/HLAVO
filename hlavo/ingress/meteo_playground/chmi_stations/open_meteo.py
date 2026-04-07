import logging
from datetime import date, datetime
import json
from pathlib import Path

import polars as pl
import requests

LOGGER = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]


def _normalize_date(value):
    """
    Convert supported date-like inputs to the YYYY-MM-DD string expected by Open-Meteo.
    """
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    assert isinstance(value, str), f"Unsupported date value: {value!r}"
    return value[:10]


def _download_open_meteo_json(latitude, longitude, start, end, *, hourly_variables=None):
    """
    Download Open-Meteo archive JSON payload for one location and time interval.
    """
    requested_variables = OPEN_METEO_HOURLY_VARIABLES if hourly_variables is None else hourly_variables
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": _normalize_date(start),
        "end_date": _normalize_date(end),
        "hourly": ",".join(requested_variables),
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    payload["_requested_hourly_variables"] = requested_variables
    return payload


def open_meteo(latitude, longitude, start, end, *, json_path=None, hourly_variables=None, refresh=False):
    """
    Download hourly historical Open-Meteo data for one location, optionally cache the raw
    JSON payload, and return it as a polars dataframe.
    """
    requested_variables = OPEN_METEO_HOURLY_VARIABLES if hourly_variables is None else hourly_variables
    cache_path = None if json_path is None else Path(json_path)

    if cache_path is not None and cache_path.exists() and not refresh:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = _download_open_meteo_json(
            latitude=latitude,
            longitude=longitude,
            start=start,
            end=end,
            hourly_variables=requested_variables,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)

    hourly_data = payload["hourly"]

    missing_variables = [variable for variable in requested_variables if variable not in hourly_data]
    assert not missing_variables, f"Open-Meteo response missing variables: {missing_variables}"

    frame = pl.DataFrame(
        {
            "date_time": hourly_data["time"],
            **{variable: hourly_data[variable] for variable in requested_variables},
        }
    ).with_columns(
        pl.col("date_time")
        .str.to_datetime(format="%Y-%m-%dT%H:%M", strict=False)
        .dt.replace_time_zone("UTC")
    )

    LOGGER.info(
        "Downloaded %s hourly Open-Meteo records for latitude=%s longitude=%s.",
        frame.height,
        latitude,
        longitude,
    )
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = open_meteo(50.83897, 14.87175, "2024-01-01", "2024-01-03")
    print(data)
