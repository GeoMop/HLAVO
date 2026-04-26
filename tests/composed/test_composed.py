from __future__ import annotations

from pathlib import Path

from dask.distributed import Client, LocalCluster

from hlavo.composed import model_composed

TESTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TESTS_DIR / "test_composed_config.yaml"


def test_setup_models_uses_real_dask_queues_with_config_mocks(tmp_path):
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
    client = Client(cluster)

    try:
        final_state = model_composed.setup_models(work_dir=work_dir, config_path=CONFIG_PATH, client=client)
    finally:
        client.close()
        cluster.close()

    assert final_state == 1.0
