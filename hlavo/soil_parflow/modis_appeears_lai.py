#!/usr/bin/env python3
"""Fetch MODIS LAI time series for a point through NASA AppEEARS.

This script submits a point request for the Aqua MODIS product MYD15A2H.061,
waits for processing to finish, downloads the AppEEARS CSV result, and writes
out a simplified LAI time series CSV.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"
PRODUCT = "MYD15A2H.061"
LAYER = "Lai_500m"
DATE_INPUT_FORMATS = ("%Y-%m-%d", "%m-%d-%Y")


class AppEEARSError(RuntimeError):
    """Raised when an AppEEARS API call fails."""


@dataclass
class DownloadedOutputs:
    task_id: str
    raw_csv: Path
    simplified_csv: Path
    bundle_manifest: Path
    task_metadata: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch MODIS LAI for a point using the NASA AppEEARS API "
            f"({PRODUCT} / {LAYER})."
        )
    )
    parser.add_argument("--latitude", type=float, required=True, help="Point latitude in decimal degrees.")
    parser.add_argument("--longitude", type=float, required=True, help="Point longitude in decimal degrees.")
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD or MM-DD-YYYY format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD or MM-DD-YYYY format.",
    )
    parser.add_argument("--task-name", help="Optional AppEEARS task name.")
    parser.add_argument("--point-id", help="Optional identifier for the requested coordinate.")
    parser.add_argument("--category", help="Optional category for the requested coordinate.")
    parser.add_argument(
        "--username",
        default=os.getenv("APPEEARS_USERNAME") or os.getenv("EARTHDATA_USERNAME"),
        help="NASA Earthdata username. Defaults to APPEEARS_USERNAME or EARTHDATA_USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("APPEEARS_PASSWORD") or os.getenv("EARTHDATA_PASSWORD"),
        help="NASA Earthdata password. Defaults to APPEEARS_PASSWORD or EARTHDATA_PASSWORD.",
    )
    parser.add_argument(
        "--output-dir",
        default="appeears_myd15a2h_lai",
        help="Directory where outputs should be written.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Polling interval in seconds while waiting for processing.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Maximum total wait time in seconds.",
    )
    return parser.parse_args()


def normalize_date(date_str: str) -> tuple[str, str]:
    for date_format in DATE_INPUT_FORMATS:
        try:
            parsed = datetime.strptime(date_str, date_format)
            return parsed.strftime("%m-%d-%Y"), parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"Unsupported date format: {date_str!r}. Use YYYY-MM-DD or MM-DD-YYYY."
    )


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "appeears_task"


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def ensure_credentials(username: str | None, password: str | None) -> tuple[str, str]:
    if not username:
        username = input("Earthdata username: ").strip()
    if not password:
        password = getpass.getpass("Earthdata password: ")
    if not username or not password:
        raise AppEEARSError("Earthdata username and password are required.")
    return username, password


class AppEEARSClient:
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.token: str | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        expected_status: int | tuple[int, ...] = (200,),
        allow_redirects: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        if isinstance(expected_status, int):
            expected = {expected_status}
        else:
            expected = set(expected_status)

        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            allow_redirects=allow_redirects,
            timeout=120,
            **kwargs,
        )

        if response.status_code not in expected:
            message = f"AppEEARS API {method} {path} failed with HTTP {response.status_code}"
            try:
                payload = response.json()
            except ValueError:
                payload = response.text.strip()
            if payload:
                message = f"{message}: {payload}"
            raise AppEEARSError(message)
        return response

    def login(self, username: str, password: str) -> None:
        response = self.request(
            "POST",
            "/login",
            expected_status=200,
            auth=(username, password),
        )
        payload = response.json()
        token = payload.get("token")
        if not token:
            raise AppEEARSError(f"Login succeeded but no token was returned: {payload}")
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def logout(self) -> None:
        if not self.token:
            return
        try:
            self.request("POST", "/logout", expected_status=(204, 403))
        finally:
            self.token = None
            self.session.headers.pop("Authorization", None)

    def submit_point_task(
        self,
        *,
        task_name: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        point_id: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        coordinate: dict[str, Any] = {"latitude": latitude, "longitude": longitude}
        if point_id:
            coordinate["id"] = point_id
        if category:
            coordinate["category"] = category

        task = {
            "task_type": "point",
            "task_name": task_name,
            "params": {
                "dates": [{"startDate": start_date, "endDate": end_date}],
                "layers": [{"product": PRODUCT, "layer": LAYER}],
                "coordinates": [coordinate],
            },
        }
        response = self.request("POST", "/task", expected_status=202, json=task)
        return response.json()

    def get_task_status(self, task_id: str) -> list[dict[str, Any]] | dict[str, Any]:
        response = self.request(
            "GET",
            f"/status/{task_id}",
            expected_status=(200, 303),
            allow_redirects=False,
        )
        if response.status_code == 303:
            return {"task_id": task_id, "status": "done"}
        return response.json()

    def wait_for_task(
        self,
        task_id: str,
        *,
        poll_seconds: int,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_payload = self.get_task_status(task_id)
            if isinstance(status_payload, dict):
                if status_payload.get("status") == "done":
                    return status_payload
                if status_payload.get("status") == "error":
                    raise AppEEARSError(f"Task {task_id} failed: {status_payload}")
            elif status_payload:
                status_item = status_payload[0]
                progress = status_item.get("progress", {})
                summary = progress.get("summary")
                if summary is not None:
                    print(f"Task {task_id}: {summary}% complete", file=sys.stderr)
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Timed out after {timeout_seconds} seconds waiting for task {task_id}."
        )

    def get_bundle(self, task_id: str) -> dict[str, Any]:
        response = self.request("GET", f"/bundle/{task_id}", expected_status=200)
        return response.json()

    def download_bundle_file(self, task_id: str, file_id: str, destination: Path) -> None:
        response = self.request(
            "GET",
            f"/bundle/{task_id}/{file_id}",
            expected_status=200,
            stream=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def select_results_csv(bundle: dict[str, Any]) -> dict[str, Any]:
    files = bundle.get("files", [])
    preferred_name = f"{PRODUCT.replace('.', '-')}-results.csv"
    for file_info in files:
        filename = file_info.get("file_name", "")
        if filename.endswith(preferred_name):
            return file_info
    for file_info in files:
        filename = file_info.get("file_name", "")
        if filename.endswith("-results.csv"):
            return file_info
    raise AppEEARSError(f"No point results CSV found in bundle: {bundle}")


def parse_lai_results(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AppEEARSError(f"Results CSV is missing a header row: {csv_path}")

        lai_field = next((name for name in reader.fieldnames if name.endswith("_Lai_500m")), None)
        if not lai_field:
            raise AppEEARSError(f"Could not find a Lai_500m column in {csv_path}")

        tile_field = next((name for name in reader.fieldnames if name.endswith("Tile")), "MODIS_Tile")
        line_field = next((name for name in reader.fieldnames if name.endswith("Line_Y_500m")), None)
        sample_field = next((name for name in reader.fieldnames if name.endswith("Sample_X_500m")), None)
        qc_field = next((name for name in reader.fieldnames if name.endswith("_FparLai_QC")), None)
        qc_desc_field = next(
            (name for name in reader.fieldnames if name.endswith("_FparLai_QC_MODLAND_Description")),
            None,
        )
        scf_desc_field = next(
            (name for name in reader.fieldnames if name.endswith("_FparLai_QC_SCF_QC_Description")),
            None,
        )
        cloud_desc_field = next(
            (name for name in reader.fieldnames if name.endswith("_FparLai_QC_CloudState_Description")),
            None,
        )

        simplified_rows: list[dict[str, Any]] = []
        for row in reader:
            simplified_rows.append(
                {
                    "date": row.get("Date", ""),
                    "latitude": parse_optional_float(row.get("Latitude")),
                    "longitude": parse_optional_float(row.get("Longitude")),
                    "lai": parse_optional_float(row.get(lai_field)),
                    "modis_tile": row.get(tile_field, ""),
                    "line_y_500m": parse_optional_float(row.get(line_field)) if line_field else None,
                    "sample_x_500m": parse_optional_float(row.get(sample_field)) if sample_field else None,
                    "fparlai_qc": parse_optional_float(row.get(qc_field)) if qc_field else None,
                    "modland_description": row.get(qc_desc_field, "") if qc_desc_field else "",
                    "retrieval_description": row.get(scf_desc_field, "") if scf_desc_field else "",
                    "cloud_state_description": row.get(cloud_desc_field, "") if cloud_desc_field else "",
                }
            )
    return simplified_rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise AppEEARSError("No rows were parsed from the AppEEARS results CSV.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def fetch_lai(args: argparse.Namespace) -> DownloadedOutputs:
    start_mmddyyyy, start_iso = normalize_date(args.start_date)
    end_mmddyyyy, end_iso = normalize_date(args.end_date)
    task_name = args.task_name or (
        f"MYD15A2H_LAI_{start_iso}_{end_iso}_{args.latitude:.4f}_{args.longitude:.4f}"
    )
    task_name = sanitize_name(task_name)
    output_dir = Path(args.output_dir) / task_name
    username, password = ensure_credentials(args.username, args.password)

    client = AppEEARSClient()
    try:
        client.login(username, password)
        task_response = client.submit_point_task(
            task_name=task_name,
            latitude=args.latitude,
            longitude=args.longitude,
            start_date=start_mmddyyyy,
            end_date=end_mmddyyyy,
            point_id=args.point_id,
            category=args.category,
        )
        task_id = task_response["task_id"]
        print(f"Submitted task {task_id}", file=sys.stderr)

        final_status = client.wait_for_task(
            task_id,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"Task {task_id} finished with status={final_status.get('status', 'done')}", file=sys.stderr)

        bundle = client.get_bundle(task_id)
        results_file = select_results_csv(bundle)

        raw_csv_path = output_dir / results_file["file_name"]
        simplified_csv_path = output_dir / "lai_timeseries.csv"
        bundle_manifest_path = output_dir / "bundle.json"
        task_metadata_path = output_dir / "task.json"

        client.download_bundle_file(task_id, results_file["file_id"], raw_csv_path)
        simplified_rows = parse_lai_results(raw_csv_path)
        write_csv(simplified_rows, simplified_csv_path)
        save_json(bundle, bundle_manifest_path)
        save_json(task_response, task_metadata_path)

        return DownloadedOutputs(
            task_id=task_id,
            raw_csv=raw_csv_path,
            simplified_csv=simplified_csv_path,
            bundle_manifest=bundle_manifest_path,
            task_metadata=task_metadata_path,
        )
    finally:
        client.logout()


def main() -> int:
    args = parse_args()
    try:
        outputs = fetch_lai(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Task ID: {outputs.task_id}")
    print(f"Raw AppEEARS CSV: {outputs.raw_csv}")
    print(f"Simplified LAI CSV: {outputs.simplified_csv}")
    print(f"Bundle manifest: {outputs.bundle_manifest}")
    print(f"Task metadata: {outputs.task_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
