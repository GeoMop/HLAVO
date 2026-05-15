from __future__ import annotations

from pathlib import Path

from dask.distributed import Client, LocalCluster
import yaml

from hlavo.composed import model_composed


def _write_config(config_path: Path) -> None:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-06T00:00:00",
        "end_datetime": "2025-03-07T00:00:00",
        "model_3d": {
            "backend_class_name": "Model3DBackendMock",
            "common": {
                "name": "uhelna",
                "time_step_hours": 12.0,
            },
        },
        "model_1d": {
            "site_ids": [0, 1],
            "data_class_name": "Model1DDataMock",
            "kalman_class_name": "KalmanMock",
            "mock_velocity": 3.0e-4,
            "moisture_sigma": 0.05,
            "sites": [
                {"longitude": 14.88, "latitude": 50.86},
                {"longitude": 14.90, "latitude": 50.88},
            ],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_setup_models_uses_real_dask_queues_with_config_mocks(tmp_path: Path) -> None:
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    config_path = tmp_path / "test_composed_config.yaml"

    _write_config(config_path)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)

    try:
        final_state = model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=client)
    finally:
        client.close()
        cluster.close()

    assert final_state == {
        0: 3.0e-4,
        1: 3.0e-4,
    }
