from __future__ import annotations

import json
from pathlib import Path

import yaml
from dask.distributed import Client, LocalCluster

from hlavo.composed import model_composed

LIBMF6 = Path("/home/hlavo/miniconda3/envs/hlavo/lib/libmf6.so")


def test_modflowapi_backend_runs_inside_composed_loop(tmp_path):
    assert LIBMF6.exists(), f"MODFLOW API library not found: {LIBMF6}"
    config_path = _write_config(tmp_path)
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
    client = Client(cluster)
    try:
        final_state = model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=client)
    finally:
        client.close()
        cluster.close()

    assert final_state == {0: -60.0}
    rows = [
        json.loads(line)
        for line in (work_dir / "api_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert {row["node"] for row in rows} == {"site_prediction"}


def _write_config(tmp_path: Path) -> Path:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-01T00:00:00",
        "end_datetime": "2025-03-03T00:00:00",
        "model_3d": {
            "common": {
                "name": "uhelna",
                "backend_class_name": "Model3DAPI",
                "sim_name": "api_cube",
                "model_name": "GWF",
                "recharge_package": "RCHA",
                "simulation_days": 2.0,
                "time_step_hours": 24.0,
                "initial_head": -60.0,
                "top": 0.0,
                "bottom": -100.0,
                "lib_path": str(LIBMF6),
                "writer": {
                    "class_name": "FilePredictionWriter",
                    "file_name": "api_predictions.jsonl",
                },
            },
        },
        "model_1d": {
            "kalman_class_name": "KalmanMock",
            "mock_velocity": 1.0e-8,
            "moisture_sigma": 0.05,
            "sites": [
                {"longitude": 14.88, "latitude": 50.86},
            ],
        },
    }
    config_path = tmp_path / "modflowapi_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
