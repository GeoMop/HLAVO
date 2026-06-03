from __future__ import annotations

import shutil
from pathlib import Path

from dask.distributed import Client, LocalCluster
import pandas as pd
import yaml
import zarr_fuse as zf

from hlavo.composed import model_composed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_CONFIG = REPO_ROOT / "runs" / "composed_1d_only" / "composed_config.yaml"
PROFILE_SCHEMA = SCRIPT_DIR / "profile_schema.yaml"
SURFACE_SCHEMA = SCRIPT_DIR / "chmi_stations_schema.yaml"
PROFILE_STORE = SCRIPT_DIR / "profiles.zarr"
SURFACE_STORE = SCRIPT_DIR / "chmi_stations.zarr"


def _reset_store(path: Path) -> None:
    assert path.parent == SCRIPT_DIR
    if path.exists():
        shutil.rmtree(path)


def _copy_schema(schema_key: str, destination: Path, store_path: Path) -> None:
    with RUN_CONFIG.open("r", encoding="utf-8") as stream:
        run_config = yaml.safe_load(stream)

    schema_file = run_config["model_1d"]["schema_files"][schema_key]
    source = (RUN_CONFIG.parent / schema_file).resolve()
    shutil.copyfile(source, destination)

    with destination.open("r", encoding="utf-8") as stream:
        schema = yaml.safe_load(stream)
    schema["ATTRS"]["STORE_URL"] = str(store_path.relative_to(REPO_ROOT))
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(schema, stream, sort_keys=False)


def _copy_schemas() -> None:
    _copy_schema("profiles", PROFILE_SCHEMA, PROFILE_STORE)
    _copy_schema("surface", SURFACE_SCHEMA, SURFACE_STORE)


def _make_profile_store() -> None:
    _reset_store(PROFILE_STORE)
    node = zf.open_store(PROFILE_SCHEMA)["Uhelna"]["profiles"]
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


def _make_surface_store() -> None:
    _reset_store(SURFACE_STORE)
    node = zf.open_store(SURFACE_SCHEMA)["Uhelna"]["parflow"]["version_01"]
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


def _make_zarr_stores() -> None:
    _copy_schemas()
    _make_profile_store()
    _make_surface_store()


def test_setup_models_uses_real_dask_queues_with_config_mocks(tmp_path: Path) -> None:
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    config_path = SCRIPT_DIR / "test_composed_config.yaml"
    _make_zarr_stores()

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)

    try:
        final_state = model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=client)
    finally:
        client.close()
        cluster.close()

    assert final_state == {
        1: 3.0e-4,
    }
