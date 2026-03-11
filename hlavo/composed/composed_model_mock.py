#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from dask.distributed import Client, LocalCluster, Queue, get_client

from hlavo.composed.model_1d_worker import model1d_worker_entry
from hlavo.composed.model_3d_driver import Model3D

LOGGER = logging.getLogger(__name__)


def setup_models(
    n_1d: int,
    start_time: np.datetime64,
    t_end: np.datetime64,
    base_dt: np.timedelta64,
    work_dir: Path,
    kalman_config: Path,
) -> float:
    client = get_client()
    run_id = uuid4().hex[:12]
    queue_names_3d_to_1d = []
    futures_1d = []

    queue_name_1d_to_3d = f"q-1d-to-3d-{run_id}"
    Queue(queue_name_1d_to_3d, client=client)

    for i in range(n_1d):
        q_name_3d_to_1d = f"q-3d-to-1d-{i}-{run_id}"
        Queue(q_name_3d_to_1d, client=client)
        queue_names_3d_to_1d.append(q_name_3d_to_1d)

        future_1d = client.submit(
            model1d_worker_entry,
            i,
            t_end,
            q_name_3d_to_1d,
            queue_name_1d_to_3d,
            work_dir,
            kalman_config,
            pure=False,
        )
        futures_1d.append(future_1d)
        LOGGER.info("[SETUP] Submitted Model1D idx=%s", i)

    model_3d = Model3D(n_1d=n_1d, initial_time=start_time, base_dt=base_dt)
    final_state_3d = model_3d.run_loop(
        t_end=t_end,
        queue_names_out_to_1d=queue_names_3d_to_1d,
        queue_name_in_from_1d=queue_name_1d_to_3d,
    )

    LOGGER.info("[SETUP] Waiting for all 1D models to finish...")
    results_1d = [future.result() for future in futures_1d]
    LOGGER.info("[SETUP] 1D model results: %s", results_1d)
    return final_state_3d


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", help="Path to work dir")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument(
        "--start-time",
        default="2020-01-01T00:00:00.000",
        help="Simulation start as numpy datetime64-compatible string.",
    )
    parser.add_argument("--n-steps", type=int, default=5, help="Number of 3D timesteps to execute.")
    parser.add_argument("--dt-minutes", type=int, default=60, help="Length of base timestep in minutes.")
    parser.add_argument("--n-1d", type=int, default=3, help="Count of 1D models.")
    args = parser.parse_args(argv)

    if args.n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if args.dt_minutes <= 0:
        raise ValueError("dt_minutes must be > 0")
    if args.n_1d <= 0:
        raise ValueError("n_1d must be > 0")

    work_dir = Path(args.work_dir)
    kalman_config = Path(args.config_file).resolve()
    start_time = np.datetime64(args.start_time, "ms")
    dt = np.timedelta64(args.dt_minutes, "m")
    t_end = start_time + args.n_steps * dt

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    try:
        final_state = setup_models(args.n_1d, start_time, t_end, dt, work_dir, kalman_config)
        LOGGER.info("[MAIN] Final 3D state: %s", final_state)
    finally:
        client.close()
        cluster.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

