from __future__ import annotations

import logging
from pathlib import Path

import yaml
from dask.distributed import Client, LocalCluster, Queue

from hlavo.composed.common_data import ComposedData
from hlavo.composed.worker_1d import Model1DLocation, model1d_worker_entry
from hlavo.composed.model_3d import Model3D
from hlavo.deep_model.coupled_runtime import CoupledModel3DConfig as Model3DConfig
from hlavo.misc.config import load_config

LOG = logging.getLogger(__name__)


def setup_models(work_dir, config_path, client):
    work_dir = Path(work_dir).resolve()
    config_path = Path(config_path).resolve()

    config_data, _ = load_config(config_path)
    composed = ComposedData.from_config(work_dir, config_data, config_path)
    locations_1d = config_data["model_1d"]["sites"]

    queue_names_3d_to_1d = []
    futures_1d = []

    queue_name_1d_to_3d = "q-1d-to-3d"
    Queue(queue_name_1d_to_3d, client=client)


    model_3d = Model3D(composed, config_data['model_3d'], locations_1d)
    #t_end = model_3d.resolve_t_end()

    for i in range(len(locations_1d)):
        q_name_3d_to_1d = f"q-3d-to-1d-{i}"
        Queue(q_name_3d_to_1d, client=client)
        queue_names_3d_to_1d.append(q_name_3d_to_1d)

        fut = client.submit(
            model1d_worker_entry,
            composed,
            i,
            locations_1d[i],
            config_data['model_1d'],
            q_name_3d_to_1d,
            queue_name_1d_to_3d,
        )
        futures_1d.append(fut)
        LOG.info("[SETUP] Submitted Model1D idx=%s", i)

    final_state_3d = model_3d.run_loop(
        queue_names_out_to_1d=queue_names_3d_to_1d,
        queue_name_in_from_1d=queue_name_1d_to_3d,
    )

    LOG.info("[SETUP] Waiting for all 1D models to finish...")
    results_1d = [f.result() for f in futures_1d]
    LOG.info("[SETUP] 1D model results: %s", results_1d)

    return final_state_3d


# def _parse_locations(config_data):
#     model_1d_cfg = config_data.get("model_1d", {})
#     assert isinstance(model_1d_cfg, dict), "model_1d must be a mapping"
#     raw_locations = model_1d_cfg.get("sites", [])
#     assert isinstance(raw_locations, list), "model_1d.sites must be a list"
#     assert raw_locations, "model_1d.sites must not be empty"
#
#     locations = []
#     for idx, item in enumerate(raw_locations):
#         assert isinstance(item, dict), "Each model_1d item must be a mapping"
#         locations.append(
#             Model1DLocation(
#                 idx=idx,
#                 longitude=float(item["longitude"]),
#                 latitude=float(item["latitude"]),
#             )
#         )
#     return locations


def run_simulation(work_dir: Path, config_path: Path) -> float:
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    try:
        final_state = setup_models(work_dir, config_path, client)
        LOG.info("[MAIN] Final 3D time: %s", final_state)
        return float(final_state)
    finally:
        client.close()
        cluster.close()
