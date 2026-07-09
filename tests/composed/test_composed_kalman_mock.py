from __future__ import annotations

import shutil
from pathlib import Path

from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import yaml
import zarr_fuse as zf

from hlavo.composed import model_composed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_CONFIG = REPO_ROOT / "runs" / "composed_1d_only" / "composed_config.yaml"
CONFIG_PATH = SCRIPT_DIR / "test_composed_kalman_mock_config.yaml"


def _copy_schema(schema_key: str, destination: Path, store_path: Path) -> None:
    with RUN_CONFIG.open("r", encoding="utf-8") as stream:
        run_config = yaml.safe_load(stream)

    schema_file = run_config["model_1d"]["schema_files"][schema_key]
    source = (RUN_CONFIG.parent / schema_file).resolve()
    shutil.copyfile(source, destination)

    with destination.open("r", encoding="utf-8") as stream:
        schema = yaml.safe_load(stream)
    schema["ATTRS"]["STORE_URL"] = f"file://{store_path}"
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(schema, stream, sort_keys=False)


def _make_profile_store(schema_path: Path) -> None:
    node = zf.open_store(schema_path)["Uhelna"]["profiles"]
    rows = []
    for date_time in pd.to_datetime(
        ["2025-03-06 00:00", "2025-03-06 12:00", "2025-03-07 00:00"]
    ):
        for depth_level, sensor_depth in enumerate([0.1, 0.2, 0.6]):
            rows.append(
                {
                    "date_time": date_time,
                    "site_id": 1,
                    "depth_level": depth_level,
                    "probe_model": "PR2",
                    "T_sensor": 10.0 + depth_level,
                    "T_probe": 11.0,
                    "moisture": 0.2 + 0.01 * depth_level,
                    "permeability": 1.0,
                    "longitude": 14.889853,
                    "latitude": 50.863565,
                    "probe_id": "PR-1",
                    "site_status": 1,
                    "sensor_depth": sensor_depth,
                    "manufacture_id": "mock-1",
                }
            )
    node.update(pd.DataFrame(rows))


def _make_surface_store(schema_path: Path) -> None:
    node = zf.open_store(schema_path)["Uhelna"]["parflow"]["version_01"]
    rows = []
    for date_time in pd.date_range("2025-03-06 00:00", "2025-03-07 00:00", freq="12h"):
        rows.append(
            {
                "date_time": date_time,
                "site_id": 1,
                "APCP": 1.0e-6,
                "Temp": 273.15,
                "UGRD": 0.0,
                "VGRD": 0.0,
                "Press": 100000.0,
                "SPFH": 0.005,
                "DSWR": 150.0,
                "DLWR": 250.0,
                "longitude": 14.889853,
                "latitude": 50.863565,
                "elevation": 300.0,
            }
        )
    node.update(pd.DataFrame(rows))


def _make_runtime_config(tmp_path: Path) -> Path:
    profiles_store = tmp_path / "profiles.zarr"
    surface_store = tmp_path / "chmi_stations.zarr"
    profiles_schema = tmp_path / "profile_schema.yaml"
    surface_schema = tmp_path / "chmi_stations_schema.yaml"

    _copy_schema("profiles", profiles_schema, profiles_store)
    _copy_schema("surface", surface_schema, surface_store)
    _make_profile_store(profiles_schema)
    _make_surface_store(surface_schema)

    config_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime_config_path = tmp_path / "test_composed_kalman_mock_runtime_config.yaml"
    config_data["model_1d"]["schema_files"] = {
        "profiles": profiles_schema.name,
        "surface": surface_schema.name,
    }
    runtime_config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_config_path


def test_setup_models_uses_real_dask_queues_with_kalman_mock_zarr_store(tmp_path: Path) -> None:
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    runtime_config_path = _make_runtime_config(tmp_path)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)

    try:
        final_state = model_composed.setup_models(
            work_dir=work_dir,
            config_path=runtime_config_path,
            client=client,
        )
    finally:
        client.close()
        cluster.close()

    assert final_state == np.datetime64("2025-03-07T00:00:00")
