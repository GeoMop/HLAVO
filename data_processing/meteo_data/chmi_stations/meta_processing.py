import json
import math
from collections import defaultdict

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def get_data_block(doc):
    """
    For CHMI-style JSON:

    {
      "data": {
        "type": "DataCollection",
        "data": {
          "header": "...",
          "values": [...]
        }
      }
    }

    or sometimes:

    {
      "type": "DataCollection",
      "data": {
        "header": "...",
        "values": [...]
      }
    }

    return the dict that has 'header' and 'values'.
    """
    # case 1: top level is DataCollection
    if "header" in doc and "values" in doc:
        return doc

    # case 2: outer 'data' with inner 'data'
    data = doc.get("data")
    if isinstance(data, dict):
        if "header" in data and "values" in data:
            return data
        inner = data.get("data")
        if isinstance(inner, dict) and "header" in inner and "values" in inner:
            return inner

    raise RuntimeError("Could not find 'header'/'values' block in JSON document.")


def safe_float(x):
    """Convert value to float or return None on empty/invalid."""
    if x in ("", None):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (lat, lon) in decimal degrees.
    Returns distance in kilometres.
    """
    # convert to radians
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)

    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1

    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    R = 6371.0  # Earth radius in km
    return R * c


# -----------------------------
# Subscript 1: meta2 processing
# -----------------------------
def collect_quantity_definitions(meta2_path):
    """
    Read a JSON DataCollection 'meta2' file and, for each quantity abbreviation
    (EG_EL_ABBREVIATION), return a dict:
        { "T": {("Teplota", "°C"), ...}, ... }
    """
    with open(meta2_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    data_block = get_data_block(doc)
    header = data_block["header"].split(",")
    values = data_block["values"]

    try:
        idx_abbr = header.index("EG_EL_ABBREVIATION")
        idx_name = header.index("NAME")
        idx_unit = header.index("UN_DESCRIPTION")
    except ValueError as e:
        raise RuntimeError(f"Required column missing in meta2 header: {e}")

    mapping = defaultdict(set)

    for row in values:
        abbr = row[idx_abbr]
        name = row[idx_name]
        unit = row[idx_unit]
        mapping[abbr].add((name, unit))

    return mapping


def build_station_quantity_map(meta2_path):
    """
    From meta2, build:
      - a sorted list of all quantity abbreviations
      - a dict WSI -> set of abbreviations available at that station
    """
    with open(meta2_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    data_block = get_data_block(doc)
    header = data_block["header"].split(",")
    values = data_block["values"]

    try:
        idx_wsi = header.index("WSI")
        idx_abbr = header.index("EG_EL_ABBREVIATION")
    except ValueError as e:
        raise RuntimeError(f"Required column missing in meta2 header: {e}")

    station_to_quantities = defaultdict(set)
    all_abbrs = set()

    for row in values:
        wsi = row[idx_wsi]
        abbr = row[idx_abbr]
        station_to_quantities[wsi].add(abbr)
        all_abbrs.add(abbr)

    all_abbrs = sorted(all_abbrs)
    return all_abbrs, station_to_quantities


# -----------------------------
# Subscript 2: meta1 processing
# -----------------------------
def load_stations_meta1(meta1_path):
    """
    Read meta1 JSON with station positions.
    Returns a list of dicts with keys:
      WSI, FULL_NAME, LON, LAT, ELEVATION, BEGIN_DATE, END_DATE
    """
    with open(meta1_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    data_block = get_data_block(doc)
    header = data_block["header"].split(",")
    values = data_block["values"]

    try:
        idx_wsi = header.index("WSI")
        idx_name = header.index("FULL_NAME")
        idx_lon = header.index("GEOGR1")
        idx_lat = header.index("GEOGR2")
        idx_elv = header.index("ELEVATION")
        idx_beg = header.index("BEGIN_DATE")
        idx_end = header.index("END_DATE")
    except ValueError as e:
        raise RuntimeError(f"Required column missing in meta1 header: {e}")

    stations = []
    for row in values:
        stations.append(
            {
                "WSI": row[idx_wsi],
                "FULL_NAME": row[idx_name],
                "LON": safe_float(row[idx_lon]),
                "LAT": safe_float(row[idx_lat]),
                "ELEVATION": safe_float(row[idx_elv]),
                "BEGIN_DATE": row[idx_beg],
                "END_DATE": row[idx_end],
            }
        )

    return stations


# -----------------------------
# Subscript 3: build dataframe
# -----------------------------
def build_station_dataframe(
    stations,
    all_abbrs,
    station_to_quantities,
    center_lat,
    center_lon,
    max_distance_km,
):
    """
    Build a pandas DataFrame with columns:
      WSI, FULL_NAME, GPS, ELEVATION, DIST_KM, BEGIN_DATE, END_DATE,
      and one boolean column per quantity abbreviation.
    """
    records = []
    for st in stations:
        if st["LAT"] is None or st["LON"] is None:
            continue

        dist = haversine_km(center_lat, center_lon, st["LAT"], st["LON"])
        if dist > max_distance_km:
            continue

        wsi = st["WSI"]
        station_abbrs = station_to_quantities.get(wsi, set())

        rec = {
            "WSI": wsi,
            "FULL_NAME": st["FULL_NAME"],
            "GPS": f"{st['LAT']:.6f},{st['LON']:.6f}",
            "ELEVATION": st["ELEVATION"],
            "DIST_KM": dist,
            "BEGIN_DATE": st["BEGIN_DATE"],
            "END_DATE": st["END_DATE"],
        }

        # boolean availability per quantity
        for abbr in all_abbrs:
            rec[abbr] = abbr in station_abbrs

        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("DIST_KM").reset_index(drop=True)
    return df


# -----------------------------
# main: call subscripts
# -----------------------------
def main():
    # --- parameters you can tweak ---
    meta1_path = "meta1.json"
    meta2_path = "meta2.json"
    output_csv = "stations_nearby.csv"
    quantity_defs_path = "quantity_definitions.json"  # <--- new output file

    # reference location & radius (example: somewhere near Cheb)
    center_lat = 50.8659928
    center_lon = 14.8962714
    max_distance_km = 20.0
    # -------------------------------

    # 1) (optional) collect quantity definitions (abbr -> (name, unit) pairs)
    quantity_defs = collect_quantity_definitions(meta2_path)
    print(f"Loaded {len(quantity_defs)} unique quantity abbreviations (name/unit variants).")
    
    # ---- NEW: write them out as a JSON dictionary ----
    # convert sets of tuples to a JSON-serializable structure
    serializable_defs = {
        abbr: [
            {"name": name, "unit": unit}
            for (name, unit) in sorted(pairs)
        ]
        for abbr, pairs in quantity_defs.items()
    }

    with open(quantity_defs_path, "w", encoding="utf-8") as f:
        json.dump(serializable_defs, f, ensure_ascii=False, indent=2)

    print(f"Quantity definitions written to {quantity_defs_path}")
    # 2) station -> available quantities from meta2
    all_abbrs, station_to_quantities = build_station_quantity_map(meta2_path)
    print(f"Found {len(all_abbrs)} distinct abbreviations in meta2.")

    # 3) load stations from meta1
    stations = load_stations_meta1(meta1_path)
    print(f"Loaded {len(stations)} station records from meta1.")

    # 4) build dataframe filtered by distance
    df = build_station_dataframe(
        stations,
        all_abbrs,
        station_to_quantities,
        center_lat=center_lat,
        center_lon=center_lon,
        max_distance_km=max_distance_km,
    )

    # 5) output to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Written {len(df)} stations to {output_csv}")


if __name__ == "__main__":
    main()
