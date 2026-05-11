from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml
import zarr_fuse
from dask.distributed import Client, LocalCluster

from hlavo.composed import model_composed

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
CONFIG_PATH = TESTS_DIR / "test_composed_config.yaml"


def _open_store(schema_path: Path, store_path: Path):
    return zarr_fuse.open_store(schema_path, STORE_URL=str(store_path))


def _write_profiles_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  profiles:
    VARS:
      moisture:
        type: float
        coords: ["date_time", "site_id", "depth_level"]
      longitude:
        type: float
        coords: ["date_time", "site_id"]
      latitude:
        type: float
        coords: ["date_time", "site_id"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      site_id:
        type: int32
        sorted: []
      depth_level:
        type: int32
        sorted: []
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_meteo_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  parflow:
    version_01:
      VARS:
        APCP:
          type: float
          coords: ["date_time", "site_id"]
        Temp:
          type: float
          coords: ["date_time", "site_id"]
        longitude:
          type: float
          coords: ["date_time", "site_id"]
        latitude:
          type: float
          coords: ["date_time", "site_id"]
        elevation:
          type: float
          coords: ["date_time", "site_id"]
      COORDS:
        date_time:
          unit: {{ tick: "m", tz: "UTC" }}
        site_id:
          type: int32
          sorted: []
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_wells_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  water_levels:
    VARS:
      water_level:
        type: float
        coords: ["date_time", "well_id"]
      water_depth:
        type: float
        coords: ["date_time", "well_id"]
      longitude:
        type: float
        coords: ["well_id"]
      latitude:
        type: float
        coords: ["well_id"]
      Z:
        type: float
        coords: ["well_id"]
      interval_min:
        type: float
        coords: ["well_id", "interval_num_from_top"]
      interval_max:
        type: float
        coords: ["well_id", "interval_num_from_top"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      well_id:
        type: str[16]
        sorted: []
      interval_num_from_top:
        type: str[16]
        sorted: []
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_simulation_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  well_prediction:
    VARS:
      water_level:
        type: float
        coords: ["date_time", "well_id", "calibration"]
      water_depth:
        type: float
        coords: ["date_time", "well_id", "calibration"]
      longitude:
        type: float
        coords: ["well_id"]
      latitude:
        type: float
        coords: ["well_id"]
      Z:
        type: float
        coords: ["well_id"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      well_id:
        type: str[16]
        sorted: []
      calibration:
        type: str[16]
        sorted: []
  site_prediction:
    VARS:
      velocity:
        type: float
        coords: ["date_time", "site_id", "calibration"]
      preasure_head:
        type: float
        coords: ["date_time", "site_id", "calibration"]
      longitude:
        type: float
        coords: ["site_id"]
      latitude:
        type: float
        coords: ["site_id"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      site_id:
        type: int32
        sorted: []
      calibration:
        type: str[16]
        sorted: []
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_profiles_store(schema_path: Path, store_path: Path) -> None:
    _write_profiles_schema(schema_path, store_path)
    root = _open_store(schema_path, store_path)
    rows = []
    for site_id, longitude, latitude, moisture in (
        (1, 14.88, 50.86, 0.0),
        (2, 14.90, 50.88, 1.0),
    ):
        for date_time in (
            "2025-03-01T00:00",
            "2025-03-15T00:00",
            "2025-03-31T00:00",
        ):
            for depth_level in (0, 1, 2):
                rows.append(
                    {
                        "date_time": date_time,
                        "site_id": site_id,
                        "depth_level": depth_level,
                        "moisture": moisture,
                        "longitude": longitude,
                        "latitude": latitude,
                    }
                )
    root["Uhelna"]["profiles"].update(pl.DataFrame(rows))


def _build_meteo_store(schema_path: Path, store_path: Path) -> None:
    _write_meteo_schema(schema_path, store_path)
    root = _open_store(schema_path, store_path)
    rows = []
    for site_id, longitude, latitude in (
        (1, 14.88, 50.86),
        (2, 14.90, 50.88),
    ):
        for day in range(1, 32):
            rows.append(
                {
                    "date_time": f"2025-03-{day:02d}T00:00",
                    "site_id": site_id,
                    "APCP": 2.0,
                    "Temp": 273.15,
                    "longitude": longitude,
                    "latitude": latitude,
                    "elevation": 250.0,
                }
            )
    root["Uhelna"]["parflow"]["version_01"].update(pl.DataFrame(rows))


def _build_wells_store(schema_path: Path, store_path: Path) -> None:
    _write_wells_schema(schema_path, store_path)
    root = _open_store(schema_path, store_path)
    rows = []
    for well_id, longitude, latitude, z_value in (
        ("W1", 14.881, 50.861, 280.0),
        ("W2", 14.901, 50.881, 281.0),
    ):
        rows.append(
            {
                "date_time": "2025-03-01T00:00",
                "well_id": well_id,
                "interval_num_from_top": "0",
                "water_level": 250.0,
                "water_depth": 30.0,
                "longitude": longitude,
                "latitude": latitude,
                "Z": z_value,
                "interval_min": 10.0,
                "interval_max": 30.0,
            }
        )
    root["Uhelna"]["water_levels"].update(pl.DataFrame(rows))


def _build_runtime_config(
    *,
    config_path: Path,
    profiles_schema_path: Path,
    meteo_schema_path: Path,
    wells_schema_path: Path,
    writer_class_name: str,
    writer_config: dict,
) -> None:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-01T00:00:00",
        "end_datetime": "2025-04-01T00:00:00",
        "model_3d": {
            "common": {
                "name": "uhelna",
                "backend_class_name": "Model3DDelay",
                "time_step_hours": 24.0,
                "writer_class_name": writer_class_name,
                "writer": {
                    "wells_schema_path": str(wells_schema_path),
                    **writer_config,
                },
            }
        },
        "model_1d": {
            "site_ids": [1, 2],
            "kalman_class_name": "KalmanScalingMock",
            "moisture_sigma": 0.05,
            "dry_scale": 0.1,
            "wet_scale": 0.8,
            "saturation_moisture": 1.0,
            "schema_files": {
                "profiles": str(profiles_schema_path),
                "surface": str(meteo_schema_path),
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _build_runtime_config_api(
    *,
    config_path: Path,
    wells_schema_path: Path,
    writer_class_name: str,
    writer_config: dict,
) -> None:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-01T00:00:00",
        "end_datetime": "2025-03-04T00:00:00",
        "model_3d": {
            "common": {
                "backend_class_name": "Model3DAPI",
                "time_step_hours": 24.0,
                "writer_class_name": writer_class_name,
                "writer": {
                    "wells_schema_path": str(wells_schema_path),
                    **writer_config,
                },
                "builder_class_name": "SimpleCubeModelBuilder",
                "builder": {
                    "sim_name": "cube",
                    "top": 0.0,
                    "bottom": -10.0,
                    "initial_head": 0.0,
                    "hydraulic_conductivity": 1.0,
                    "specific_storage": 1.0e-5,
                    "specific_yield": 0.1,
                },
            }
        },
        "model_1d": {
            "kalman_class_name": "KalmanMock",
            "mock_velocity": 0.2,
            "moisture_sigma": 0.05,
            "sites": [
                {"longitude": 14.88, "latitude": 50.86},
                {"longitude": 14.90, "latitude": 50.88},
            ],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_setup_models(config_path: Path, work_dir: Path):
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)
    try:
        return model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=client)
    finally:
        client.close()
        cluster.close()


def test_setup_models_uses_real_dask_queues_with_config_mocks(tmp_path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    final_state = _run_setup_models(CONFIG_PATH, work_dir)

    assert final_state == {
        0: 3.0e-4,
        1: 3.0e-4,
    }


def test_composed_delay_writes_jsonl(tmp_path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    profiles_schema_path = tmp_path / "profiles_schema.yaml"
    meteo_schema_path = tmp_path / "meteo_schema.yaml"
    wells_schema_path = tmp_path / "wells_schema.yaml"
    profiles_store_path = tmp_path / "profiles.zarr"
    meteo_store_path = tmp_path / "meteo.zarr"
    wells_store_path = tmp_path / "wells.zarr"
    output_path = work_dir / "composed_output.jsonl"
    config_path = tmp_path / "composed_file_writer.yaml"

    _build_profiles_store(profiles_schema_path, profiles_store_path)
    _build_meteo_store(meteo_schema_path, meteo_store_path)
    _build_wells_store(wells_schema_path, wells_store_path)
    _build_runtime_config(
        config_path=config_path,
        profiles_schema_path=profiles_schema_path,
        meteo_schema_path=meteo_schema_path,
        wells_schema_path=wells_schema_path,
        writer_class_name="JsonlModel3DWriter",
        writer_config={"jsonl_path": output_path.name},
    )

    final_state = _run_setup_models(config_path, work_dir)
    assert set(final_state) == {1, 2}

    site_rows = 0
    well_rows = 0
    for line in output_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["kind"] == "site_prediction":
            site_rows += 1
        if payload["kind"] == "well_prediction":
            well_rows += 1

    assert site_rows == 31 * 2
    assert well_rows == 31 * 2


def test_composed_delay_writes_zarr(tmp_path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    profiles_schema_path = tmp_path / "profiles_schema.yaml"
    meteo_schema_path = tmp_path / "meteo_schema.yaml"
    wells_schema_path = tmp_path / "wells_schema.yaml"
    simulation_schema_path = tmp_path / "simulation_schema.yaml"
    profiles_store_path = tmp_path / "profiles.zarr"
    meteo_store_path = tmp_path / "meteo.zarr"
    wells_store_path = tmp_path / "wells.zarr"
    simulation_store_path = tmp_path / "simulation.zarr"
    config_path = tmp_path / "composed_zarr_writer.yaml"

    _build_profiles_store(profiles_schema_path, profiles_store_path)
    _build_meteo_store(meteo_schema_path, meteo_store_path)
    _build_wells_store(wells_schema_path, wells_store_path)
    _write_simulation_schema(simulation_schema_path, simulation_store_path)
    _build_runtime_config(
        config_path=config_path,
        profiles_schema_path=profiles_schema_path,
        meteo_schema_path=meteo_schema_path,
        wells_schema_path=wells_schema_path,
        writer_class_name="ZarrModel3DWriter",
        writer_config={
            "simulation_schema_path": str(simulation_schema_path),
        },
    )

    final_state = _run_setup_models(config_path, work_dir)
    assert set(final_state) == {1, 2}

    root = zarr_fuse.open_store(simulation_schema_path)
    site_dataset = root["Uhelna"]["site_prediction"].dataset
    well_dataset = root["Uhelna"]["well_prediction"].dataset

    assert site_dataset.sizes["date_time"] == 31
    assert site_dataset.sizes["site_id"] == 2
    assert site_dataset.sizes["calibration"] == 1
    assert well_dataset.sizes["date_time"] == 31
    assert well_dataset.sizes["well_id"] == 2
    assert well_dataset.sizes["calibration"] == 1


def test_composed_modflowapi_backend_writes_jsonl(tmp_path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    wells_schema_path = tmp_path / "wells_schema.yaml"
    wells_store_path = tmp_path / "wells.zarr"
    output_path = work_dir / "modflowapi_output.jsonl"
    config_path = tmp_path / "composed_modflowapi_writer.yaml"

    _build_wells_store(wells_schema_path, wells_store_path)
    _build_runtime_config_api(
        config_path=config_path,
        wells_schema_path=wells_schema_path,
        writer_class_name="JsonlModel3DWriter",
        writer_config={"jsonl_path": output_path.name},
    )

    final_state = _run_setup_models(config_path, work_dir)
    assert set(final_state) == {0, 1}
    assert all(np.isfinite(list(final_state.values())))

    site_rows = 0
    well_rows = 0
    for line in output_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["kind"] == "site_prediction":
            site_rows += 1
        if payload["kind"] == "well_prediction":
            well_rows += 1

    assert site_rows == 3 * 2
    assert well_rows == 3 * 2
