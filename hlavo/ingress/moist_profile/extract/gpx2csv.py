from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

"""
Auxiliary script which converts gpx file into CSV.
Reason: We need to extract elevation of our sites from the points defined in mapy.cz
(points defined by PR on web mapy.cz, points can be downloaded into gpx file, they include probe names).
Elevations were then manually copied and updated into site_coord.csv.
"""

GPX_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GPX_PATH = SCRIPT_DIR / "export.gpx"
DEFAULT_CSV_PATH = SCRIPT_DIR / "export.csv"


def extract_waypoints(gpx_path: Path) -> list[dict[str, str | float]]:
    """
    Read GPX waypoints and return rows for CSV export.
    """
    root = ET.parse(gpx_path).getroot()
    rows = []
    for waypoint in root.findall("gpx:wpt", GPX_NAMESPACE):
        name = waypoint.findtext("gpx:name", default="", namespaces=GPX_NAMESPACE)
        elevation = waypoint.findtext("gpx:ele", default="", namespaces=GPX_NAMESPACE)
        rows.append(
            {
                "name": name,
                "latitude": float(waypoint.attrib["lat"]),
                "longitude": float(waypoint.attrib["lon"]),
                "elevation": float(elevation) if elevation else "",
            }
        )
    return rows


def write_waypoints_csv(rows: list[dict[str, str | float]], csv_path: Path) -> None:
    """
    Write waypoint rows to CSV with the expected column order.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["name", "latitude", "longitude", "elevation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = extract_waypoints(DEFAULT_GPX_PATH)
    write_waypoints_csv(rows, DEFAULT_CSV_PATH)
    print(f"Wrote {len(rows)} waypoints to {DEFAULT_CSV_PATH}")


if __name__ == "__main__":
    main()
