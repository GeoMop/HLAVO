"""Create local input ZARR stores for the run `runs/composed_3d_writer` (no S3 access needed).

Synthetic data for March 2025 and the 1D sites 1 and 2:
- profiles:  soil moisture profiles every 6 h, moisture following the recent rain,
- surface:   hourly ParFlow/CLM meteo forcing (node Uhelna/parflow/version_01), a few rainy days,
- wells:     two monitoring wells with positions, used by the writer for well_prediction.
Values follow the units declared in the schemas (e.g. APCP in mm/s).

The stores and schema copies (STORE_URL pointing to the local store) are written to `inputs/`
next to this script. Paths are absolute, so run the script in the same environment as the
simulation (the HLAVO docker image); `run_simulate.sh` does it before every simulation.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import zarr_fuse as zf

LOG = logging.getLogger(__name__)

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
INPUT_DIR = RUN_DIR / "inputs"
# Source schemas; their copies in INPUT_DIR differ only by STORE_URL.
SOURCE_SCHEMAS = {
    "profiles": REPO_ROOT / "hlavo/ingress/moist_profile/profile_schema.yaml",
    "surface": REPO_ROOT / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_schema.yaml",
    "wells": REPO_ROOT / "hlavo/ingress/well_data/wells_schema.yaml",
}

# Simulated interval of config.yaml, 1D sites (site_id: longitude, latitude; synthetic positions).
START = np.datetime64("2025-03-01T00:00", "m")
END = np.datetime64("2025-04-01T00:00", "m")
SITES = {1: (14.88, 50.86), 2: (14.90, 50.88)}
# Daily precipitation totals [mm/day] for the 31 days of March; site 2 gets 80 % of site 1.
RAIN_MM_PER_DAY = np.array([
    0, 0, 4, 12, 6, 0, 0, 0, 1, 0, 0, 0, 18, 9, 2, 0,
    0, 0, 0, 3, 0, 0, 7, 15, 5, 0, 0, 0, 0, 2, 0], dtype=float)
SITE_RAIN_FACTOR = np.array([1.0, 0.8])
SENSOR_DEPTHS_M = np.array([0.1, 0.2, 0.4, 0.6])


def site_arrays():
    site_id = np.array(list(SITES), dtype=np.int32)
    lon, lat = (np.array(v) for v in zip(*SITES.values()))
    return site_id, lon, lat


def rain_mm_per_day(date_time: np.ndarray) -> np.ndarray:
    """Daily rain [mm/day] at given times, shape (n_times, n_sites)."""
    day = ((date_time - START) // np.timedelta64(1, "D")).astype(int)
    return RAIN_MM_PER_DAY[day][:, None] * SITE_RAIN_FACTOR[None, :]


def surface_dataset() -> xr.Dataset:
    date_time = np.arange(START, END, np.timedelta64(1, "h"))
    site_id, lon, lat = site_arrays()
    shape = (date_time.size, site_id.size)
    hour = (date_time - date_time.astype("datetime64[D]")) / np.timedelta64(1, "h")
    day_cycle = np.sin(np.pi * (hour - 6.0) / 12.0)[:, None]  # +1 at noon, -1 at midnight
    data = {
        "APCP": rain_mm_per_day(date_time) / 86400.0,                       # mm/s
        "Temp": np.broadcast_to(278.0 + 5.0 * day_cycle, shape),            # K
        "UGRD": np.full(shape, 2.0),                                        # m/s
        "VGRD": np.full(shape, 0.5),                                        # m/s
        "Press": np.full(shape, 98000.0),                                   # Pa
        "SPFH": np.full(shape, 0.004),                                      # kg/kg
        "DSWR": np.broadcast_to(400.0 * np.clip(day_cycle, 0.0, None), shape),  # W/m2
        "DLWR": np.full(shape, 300.0),                                      # W/m2
        "longitude": np.broadcast_to(lon, shape),
        "latitude": np.broadcast_to(lat, shape),
        "elevation": np.full(shape, 300.0),                                 # m a.s.l.
    }
    return xr.Dataset(
        data_vars={name: (("date_time", "site_id"), np.array(values)) for name, values in data.items()},
        coords={"date_time": date_time, "site_id": site_id},
    )


def profiles_dataset() -> xr.Dataset:
    date_time = np.arange(START, END, np.timedelta64(6, "h"))
    site_id, lon, lat = site_arrays()
    depth_level = np.arange(SENSOR_DEPTHS_M.size, dtype=np.int32)
    shape_2d = (date_time.size, site_id.size)
    shape_3d = (*shape_2d, depth_level.size)
    # Moisture rises with the rain of the last 3 days and decreases with depth.
    rain = rain_mm_per_day(date_time)
    kernel = np.ones(12) / 12.0  # 3 days of 6-hour samples
    recent_rain = np.stack([np.convolve(rain[:, i], kernel)[: date_time.size] for i in range(site_id.size)], axis=1)
    moisture = 0.18 + 0.2 * np.clip(recent_rain / 15.0, 0.0, 1.0)
    moisture = moisture[:, :, None] * np.linspace(1.0, 0.8, depth_level.size)[None, None, :]
    return xr.Dataset(
        data_vars={
            "T_sensor": (("date_time", "site_id", "depth_level"), np.full(shape_3d, 8.0)),
            "T_probe": (("date_time", "site_id"), np.full(shape_2d, 8.0)),
            "moisture": (("date_time", "site_id", "depth_level"), moisture),
            "permeability": (("date_time", "site_id", "depth_level"), np.full(shape_3d, 1.0)),
            "longitude": (("date_time", "site_id"), np.broadcast_to(lon, shape_2d).copy()),
            "latitude": (("date_time", "site_id"), np.broadcast_to(lat, shape_2d).copy()),
            "probe_id": (("date_time", "site_id"), np.full(shape_2d, "SYNTH", dtype="U16")),
            "site_status": (("date_time", "site_id"), np.full(shape_2d, 10, dtype=np.int32)),
            "sensor_depth": (("depth_level", "probe_model"), SENSOR_DEPTHS_M[:, None]),
            "manufacture_id": (("date_time", "site_id"), np.full(shape_2d, "SYNTH", dtype="U12")),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
            "depth_level": depth_level,
            "probe_model": np.array(["PR2"], dtype="U16"),
        },
    )


def wells_dataset() -> xr.Dataset:
    well_id = np.array(["SYN-W1", "SYN-W2"], dtype="U16")
    return xr.Dataset(
        data_vars={
            "water_level": (("date_time", "well_id"), np.full((1, 2), -60.0)),
            "water_depth": (("date_time", "well_id"), np.full((1, 2), np.nan)),
            "well_in_section_file": (("well_id",), well_id),
            "confirmed": (("well_id",), np.ones(2, dtype=np.int32)),
            "X": (("well_id",), np.array([-680000.0, -680100.0])),
            "Y": (("well_id",), np.array([-960000.0, -960100.0])),
            "longitude": (("well_id",), np.array([14.87, 14.91])),
            "latitude": (("well_id",), np.array([50.85, 50.89])),
            "Z": (("well_id",), np.array([300.0, 301.0])),
            "collector": (("well_id",), np.array(["synthetic", "synthetic"], dtype="U32")),
            "interval_min": (("well_id", "interval_num_from_top"), np.full((2, 1), 55.0)),
            "interval_max": (("well_id", "interval_num_from_top"), np.full((2, 1), 70.0)),
        },
        coords={
            "date_time": np.array([START], dtype="datetime64[m]"),
            "well_id": well_id,
            "interval_num_from_top": np.array(["0"], dtype="U16"),
        },
    )


def local_schema(key: str, store_name: str) -> Path:
    """Copy of the source schema with STORE_URL set to a fresh local store in INPUT_DIR."""
    raw = yaml.safe_load(SOURCE_SCHEMAS[key].read_text(encoding="utf-8"))
    raw["ATTRS"]["STORE_URL"] = str(INPUT_DIR / store_name)
    schema_path = INPUT_DIR / SOURCE_SCHEMAS[key].name
    schema_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shutil.rmtree(INPUT_DIR / store_name, ignore_errors=True)
    zf.remove_store(schema_path)
    return schema_path


def dense_values(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    values = {name: np.asarray(dataset.coords[name].values) for name in dataset.coords}
    values.update({name: np.asarray(dataset[name].values) for name in dataset.data_vars})
    return values


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = {
        "profiles": ("profiles.zarr", ("Uhelna", "profiles"), profiles_dataset),
        "surface": ("chmi_stations.zarr", ("Uhelna", "parflow", "version_01"), surface_dataset),
        "wells": ("wells.zarr", ("Uhelna", "water_levels"), wells_dataset),
    }
    for key, (store_name, node_path, make_dataset) in nodes.items():
        node = zf.open_store(local_schema(key, store_name))
        for name in node_path:
            node = node[name]
        dataset = make_dataset()
        node.update_dense(dense_values(dataset))
        LOG.info("Written %s: %s %s", INPUT_DIR / store_name, "/".join(node_path), dict(dataset.sizes))


if __name__ == "__main__":
    main()
