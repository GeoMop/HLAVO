from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from dask.distributed import Queue

from hlavo.composed.common_data import ComposedData
from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.kalman.model_1d import Model1D, Model1DConstantWeather
from hlavo.misc.class_resolve import resolve_named_class
from hlavo.misc.aux_zarr_fuse import load_dotenv
from hlavo.misc.logging_utils import ensure_debug_file_handler, ensure_stdout_handler, set_hlavo_loggers

LOG = logging.getLogger(__name__)


def configure_worker_logging(workdir: Path, site_id: int) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_name = f"worker_1d_site_{site_id}.log"
    # Do not append worker DEBUG logs to the main HLAVO log; parallel workers would interleave.
    log_path = (workdir / log_name).resolve()

    set_hlavo_loggers()
    ensure_debug_file_handler(log_path, "_hlavo_worker_log_handler")
    ensure_stdout_handler("_hlavo_worker_stdout_handler")
    return log_path


class Worker1D:
    def __init__(self, composed: ComposedData, site_id: int, config: dict):
        self.composed = composed
        self.site_id = site_id
        model_1d_class = resolve_named_class(
            config.get("model_1d_class_name", "Model1D"),
            (Model1D, Model1DConstantWeather),
        )
        self.model = model_1d_class.from_config(
            composed=composed,
            site_id=site_id,
            config=config,
        )

    def run_loop(self, queue_name_in: str, queue_name_out: str):
        q_in = Queue(queue_name_in)
        q_out = Queue(queue_name_out)

        current_time = np.datetime64(self.composed.start)
        t_end = np.datetime64(self.composed.end)

        while current_time < t_end:
            data_to_1d = self._receive(q_in)
            target_time = np.datetime64(data_to_1d.date_time)
            assert self.site_id == data_to_1d.site_id
            assert target_time > current_time, (
                f"Non-advancing 1D target time for site_id={self.site_id}: "
                f"current_time={current_time}, target_time={target_time}"
            )
            LOG.info(
                "[1D %s] step: %s -> %s, bottom_head=%s",
                self.site_id,
                current_time,
                target_time,
                data_to_1d.pressure_head,
            )

            velocity = self.model.step(
                current_time,
                target_time,
                data_to_1d.pressure_head,
            )
            q_out.put(
                Data1DTo3D(
                    date_time=target_time,
                    site_id=self.site_id,
                    longitude=self.model.longitude,
                    latitude=self.model.latitude,
                    velocity=velocity,
                )
            )
            current_time = target_time

        self.model.save_results()
        LOG.info("[1D %s] finished loop at t=%s (t_end=%s)", self.site_id, current_time, t_end)
        return f"1D model {self.site_id} done"

    @staticmethod
    def _receive(queue: Queue) -> Data3DTo1D:
        payload = queue.get()
        if isinstance(payload, tuple):
            assert len(payload) == 2, f"Unexpected 3D->1D tuple payload: {payload}"
            payload = payload[1]
        assert isinstance(payload, Data3DTo1D), f"Unexpected 3D->1D payload: {type(payload)}"
        return payload


def model1d_worker_entry(composed: ComposedData, site_idx, config, queue_name_in, queue_name_out):
    load_dotenv()
    site_id = int(site_idx)
    configure_worker_logging(composed.workdir, site_id)
    worker = Worker1D(
        composed=composed,
        site_id=site_idx,
        config=config,
    )
    return worker.run_loop(queue_name_in, queue_name_out)


def call_worker_from_cli():
    from hlavo.misc.aux_zarr_fuse import load_dotenv
    from hlavo.misc.config import load_config
    import sys

    cfg_path, site_id, work_dir = sys.argv[1:4]
    cfg_path = Path(cfg_path).resolve()

    load_dotenv()
    config_data, _ = load_config(cfg_path)
    composed = ComposedData.from_config(work_dir, config_data, cfg_path)
    site_id = 1

    queue_name_in = f"q-3d-to-1d-{site_id}"
    queue_name_out = "q-1d-to-3d"

    result = model1d_worker_entry(
        composed=composed,
        site_idx=site_id,
        config=config_data['model_1d'],
        queue_name_in=queue_name_in,
        queue_name_out=queue_name_out,
    )
    LOG.info("%s", result)

if __name__ == "__main__":
    call_worker_from_cli()
