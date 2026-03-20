from datetime import datetime
from typing import Any

import polars as pl


def _parse_time(time_str: str | None):
    if not time_str:
        return None
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except Exception:
        return time_str

def _normalize_yrno_forecast(json_dict: dict, dataframe_row: dict | None) -> pl.DataFrame:
    row = dataframe_row or {}

    lat = row.get("dflat")
    lon = row.get("dflon")
    grid_flag = row.get("grid")
    location = row.get("site") or row.get("location") or row.get("name")

    props = json_dict.get("properties", {})
    timeseries = props.get("timeseries", [])

    records: list[dict[str, Any]] = []

    for entry in timeseries:
        time_str = entry.get("time")
        dt = _parse_time(time_str)

        rec: dict[str, Any] = {}
        rec["date_time"] = dt

        data_block = entry.get("data", {})

        instant_details = data_block.get("instant", {}).get("details", {}) or {}
        rec.update(instant_details)

        n1h_details = data_block.get("next_1_hours", {}).get("details", {}) or {}
        for key, value in n1h_details.items():
            rec[key] = value

        if lon is not None:
            try:
                rec["longitude"] = float(lon)
            except Exception:
                rec["longitude"] = None
        else:
            rec["longitude"] = None

        if lat is not None:
            try:
                rec["latitude"] = float(lat)
            except Exception:
                rec["latitude"] = None
        else:
            rec["latitude"] = None

        if grid_flag is not None:
            rec["grid_domain"] = bool(grid_flag)
        else:
            rec["grid_domain"] = False

        if location is not None:
            rec["location"] = location

        records.append(rec)

    if not records:
        return pl.DataFrame(
            {
                "date_time": [],
                "longitude": [],
                "latitude": [],
                "grid_domain": [],
            }
        )

    return pl.DataFrame(records)

def normalize(payload: bytes, metadata: dict | None) -> pl.DataFrame:
    try:
        json_dict = pl.read_json(payload)
    except Exception:
        raise ValueError("Failed to parse JSON payload")

    dataframe_row = metadata.get("dataframe_row") if metadata else None

    return _normalize_yrno_forecast(json_dict, dataframe_row)
