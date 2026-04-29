from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client, LocalCluster
import zarr_fuse as zf

from hlavo.composed import model_composed

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_SCHEMA = REPO_ROOT / "hlavo/ingress/moist_profile/profile_schema.yaml"
SURFACE_SCHEMA = REPO_ROOT / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_schema.yaml"
WELLS_SCHEMA = REPO_ROOT / "hlavo/ingress/well_data/wells_schema.yaml"
SIMULATION_SCHEMA = REPO_ROOT / "hlavo/schemas/simulation_schema.yaml"


def test_b_file_prediction_writer_counts_entries(tmp_path):
    paths = _prepare_input_stores(tmp_path)
    config_path = _write_config(tmp_path, paths, writer_class_name="FilePredictionWriter")

    _run_composed(tmp_path, config_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / "workdir" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["node"] == "site_prediction" for row in rows) == 62
    assert sum(row["node"] == "well_prediction" for row in rows) == 62


def test_a_zarr_prediction_writer_coord_sizes(tmp_path):
    paths = _prepare_input_stores(tmp_path)
    config_path = _write_config(tmp_path, paths, writer_class_name="ZarrPredictionWriter")

    _run_composed(tmp_path, config_path)

    root = zf.open_store(_schema_copy(tmp_path, SIMULATION_SCHEMA, "simulation.zarr"))
    site_ds = root["Uhelna"]["site_prediction"].dataset
    well_ds = root["Uhelna"]["well_prediction"].dataset

    assert site_ds.sizes["date_time"] == 31
    assert site_ds.sizes["site_id"] == 2
    assert site_ds.sizes["calibration"] == 1
    assert well_ds.sizes["date_time"] == 31
    assert well_ds.sizes["well_id"] == 2
    assert well_ds.sizes["calibration"] == 1


def _run_composed(tmp_path: Path, config_path: Path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)
    try:
        return model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=client)
    finally:
        client.close()
        cluster.close()


def _write_config(tmp_path: Path, paths: dict[str, Path], writer_class_name: str) -> Path:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-01T00:00:00",
        "end_datetime": "2025-04-01T00:00:00",
        "model_3d": {
            "common": {
                "name": "uhelna",
                "backend_class_name": "Model3DDelay",
                "time_step_hours": 24.0,
                "initial_water_level": -60.0,
                "writer": {
                    "class_name": writer_class_name,
                    "file_name": "predictions.jsonl",
                    "schema_file": str(SIMULATION_SCHEMA),
                    "store_url": str(tmp_path / "simulation.zarr"),
                    "wells_schema_file": str(paths["wells_schema"]),
                    "wells_store_url": str(paths["wells_store"]),
                },
            },
        },
        "model_1d": {
            "kalman_class_name": "KalmanScalingMock",
            "precipitation_var": "APCP",
            "moisture_sigma": 0.05,
            "schema_files": {
                "profiles": str(paths["profile_schema"]),
                "surface": str(paths["surface_schema"]),
            },
            "sites": [
                {"longitude": 14.88, "latitude": 50.86},
                {"longitude": 14.90, "latitude": 50.88},
            ],
        },
    }
    config_path = tmp_path / f"config_{writer_class_name}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _prepare_input_stores(tmp_path: Path) -> dict[str, Path]:
    profile_schema = _schema_copy(tmp_path, PROFILE_SCHEMA, "profiles.zarr")
    surface_schema = _schema_copy(tmp_path, SURFACE_SCHEMA, "surface.zarr")
    wells_schema = _schema_copy(tmp_path, WELLS_SCHEMA, "wells.zarr")

    profile_store = tmp_path / "profiles.zarr"
    surface_store = tmp_path / "surface.zarr"
    wells_store = tmp_path / "wells.zarr"
    _remove_store(profile_schema, profile_store)
    _remove_store(surface_schema, surface_store)
    _remove_store(wells_schema, wells_store)

    zf.open_store(profile_schema)["Uhelna"]["profiles"].update_dense(_dataset_values(_profiles_dataset()))
    zf.open_store(surface_schema)["Uhelna"]["parflow"]["version_01"].update_dense(_dataset_values(_surface_dataset()))
    zf.open_store(wells_schema)["Uhelna"]["water_levels"].update_dense(_dataset_values(_wells_dataset()))
    return {
        "profile_schema": profile_schema,
        "surface_schema": surface_schema,
        "wells_schema": wells_schema,
        "wells_store": wells_store,
    }


def _schema_copy(tmp_path: Path, source: Path, store_name: str) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["ATTRS"]["STORE_URL"] = str(tmp_path / store_name)
    schema_path = tmp_path / source.name
    schema_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return schema_path


def _remove_store(schema_path: Path, store_path: Path) -> None:
    shutil.rmtree(store_path, ignore_errors=True)
    zf.remove_store(schema_path)


def _profiles_dataset() -> xr.Dataset:
    date_time = np.array(["2025-03-01T00:00", "2025-03-15T00:00"], dtype="datetime64[m]")
    site_id = np.array([0, 1], dtype=np.int32)
    depth_level = np.array([0, 1], dtype=np.int32)
    probe_model = np.array(["PR2"], dtype="U16")
    shape_3d = (date_time.size, site_id.size, depth_level.size)
    shape_2d = (date_time.size, site_id.size)
    return xr.Dataset(
        data_vars={
            "T_sensor": (("date_time", "site_id", "depth_level"), np.full(shape_3d, 10.0)),
            "T_probe": (("date_time", "site_id"), np.full(shape_2d, 10.0)),
            "moisture": (("date_time", "site_id", "depth_level"), np.full(shape_3d, 0.5)),
            "permeability": (("date_time", "site_id", "depth_level"), np.full(shape_3d, 1.0)),
            "longitude": (("date_time", "site_id"), np.tile([14.88, 14.90], (date_time.size, 1))),
            "latitude": (("date_time", "site_id"), np.tile([50.86, 50.88], (date_time.size, 1))),
            "probe_id": (("date_time", "site_id"), np.full(shape_2d, "P1", dtype="U16")),
            "site_status": (("date_time", "site_id"), np.ones(shape_2d, dtype=np.int32)),
            "sensor_depth": (("depth_level", "probe_model"), np.array([[0.1], [0.2]], dtype=float)),
            "manufacture_id": (("date_time", "site_id"), np.full(shape_2d, "M1", dtype="U12")),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
            "depth_level": depth_level,
            "probe_model": probe_model,
        },
    )


def _surface_dataset() -> xr.Dataset:
    date_time = np.arange(
        np.datetime64("2025-03-01T00:00", "m"),
        np.datetime64("2025-04-01T00:00", "m"),
        np.timedelta64(1, "D"),
    )
    site_id = np.array([0, 1], dtype=np.int32)
    shape = (date_time.size, site_id.size)
    data = {
        "APCP": np.full(shape, 10.0),
        "Temp": np.full(shape, 273.15),
        "UGRD": np.zeros(shape),
        "VGRD": np.zeros(shape),
        "Press": np.full(shape, 100000.0),
        "SPFH": np.full(shape, 0.001),
        "DSWR": np.zeros(shape),
        "DLWR": np.zeros(shape),
        "longitude": np.tile([14.88, 14.90], (date_time.size, 1)),
        "latitude": np.tile([50.86, 50.88], (date_time.size, 1)),
        "elevation": np.full(shape, 300.0),
    }
    return xr.Dataset(
        data_vars={name: (("date_time", "site_id"), values) for name, values in data.items()},
        coords={"date_time": date_time, "site_id": site_id},
    )


def _wells_dataset() -> xr.Dataset:
    date_time = np.array(["2025-03-01T00:00"], dtype="datetime64[m]")
    well_id = np.array(["W1", "W2"], dtype="U16")
    interval_num_from_top = np.array(["0"], dtype="U16")
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
            "collector": (("well_id",), np.array(["A", "B"], dtype="U32")),
            "interval_min": (("well_id", "interval_num_from_top"), np.full((2, 1), 1.0)),
            "interval_max": (("well_id", "interval_num_from_top"), np.full((2, 1), 2.0)),
        },
        coords={
            "date_time": date_time,
            "well_id": well_id,
            "interval_num_from_top": interval_num_from_top,
        },
    )


def _dataset_values(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    values = {name: np.asarray(dataset.coords[name].values) for name in dataset.coords}
    values.update({name: np.asarray(dataset[name].values) for name in dataset.data_vars})
    return values
