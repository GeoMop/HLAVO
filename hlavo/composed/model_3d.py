from __future__ import annotations

import logging

import numpy as np
from dask.distributed import Queue

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.deep_model.coupled_runtime import CoupledModel3DConfig as Model3DConfig, Model3DBackend

LOG = logging.getLogger(__name__)
TIME_ORIGIN = np.datetime64("2000-01-01T00:00:00", "ms")
MILLISECONDS_PER_DAY = 86_400_000.0


def model_time_days_to_datetime64(model_time_days: float) -> np.datetime64:
    milliseconds = int(round(float(model_time_days) * MILLISECONDS_PER_DAY))
    return TIME_ORIGIN + np.timedelta64(milliseconds, "ms")


def datetime64_to_model_time_days(date_time: np.datetime64) -> float:
    delta_ms = (np.datetime64(date_time, "ms") - TIME_ORIGIN) / np.timedelta64(1, "ms")
    return float(delta_ms) / MILLISECONDS_PER_DAY


Model3DBackendMock = Model3DBackend


class Model3D:
    def __init__(self, n_1d, model_3d_cfg: Model3DConfig, locations_1d, initial_time=0.0):
        self.n_1d = n_1d
        self.cfg = model_3d_cfg
        self.locations_1d = locations_1d
        self.time: float = float(initial_time)
        self.backend = Model3DBackendMock(model_3d_cfg=model_3d_cfg, locations_1d=locations_1d)

    def resolve_t_end(self) -> float:
        return self.cfg.common.resolve_t_end()

    def run_loop(
        self,
        t_interval: float | tuple[np.datetime64, np.datetime64],
        queue_names_out_to_1d: list[str],
        queue_name_in_from_1d: str,
    ):
        if isinstance(t_interval, tuple):
            _, t_end_raw = t_interval
            t_end = datetime64_to_model_time_days(t_end_raw)
        else:
            t_end = float(t_interval)
        q_3d_to_1d = [Queue(name) for name in queue_names_out_to_1d]
        q_1d_to_3d = Queue(queue_name_in_from_1d)

        self.backend.build_cell_assignment()
        initial_heads = self.backend.initial_heads_to_1d()
        for i in range(self.n_1d):
            msg_out = Data3DTo1D(
                date_time=model_time_days_to_datetime64(self.time),
                site_id=i,
                pressure_head=float(initial_heads[i]),
            )
            q_3d_to_1d[i].put(msg_out)
            LOG.info("[3D] startup push -> 1D %s: date_time=%s, head=%s", i, msg_out.date_time, initial_heads[i])

        while self.time < t_end:
            dt = self.backend.choose_dt(self.time, t_end)
            if dt <= 0.0:
                LOG.info("[3D] dt <= 0, stopping to avoid infinite loop.")
                break

            target_time = self.time + dt
            LOG.info("[3D] === Step: t=%s -> t=%s ===", self.time, target_time)

            contributions = [None] * self.n_1d
            received = 0
            while received < self.n_1d:
                msg_in = q_1d_to_3d.get()
                assert isinstance(msg_in, Data1DTo3D), f"Unexpected 1D->3D payload: {type(msg_in)}"
                idx = int(msg_in.site_id)
                assert 0 <= idx < self.n_1d, f"site_id out of range: {idx}"
                LOG.info("[3D] received from 1D %s: date_time=%s, recharge=%s", idx, msg_in.date_time, msg_in.velocity)
                contributions[idx] = float(msg_in.velocity)
                received += 1

            heads_to_1d = self.backend.model_step(dt, contributions)
            for i in range(self.n_1d):
                msg_out = Data3DTo1D(
                    date_time=model_time_days_to_datetime64(target_time),
                    site_id=i,
                    pressure_head=float(heads_to_1d[i]),
                )
                q_3d_to_1d[i].put(msg_out)
                LOG.info("[3D] send head -> 1D %s: date_time=%s, head=%s", i, msg_out.date_time, heads_to_1d[i])

            self.time = target_time

        LOG.info("[3D] finished time loop at t=%s (t_end=%s)", self.time, t_end)
        return self.time
