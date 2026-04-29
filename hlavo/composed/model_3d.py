from __future__ import annotations
from typing import *

import logging

from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)

import numpy as np
from dask.distributed import Queue

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
import hlavo.deep_model.model_3d_cfg as cfg3d
from hlavo.deep_model.coupled_runtime import Model3DBackend
from hlavo.misc.class_resolve import resolve_named_class

TIME_ORIGIN = np.datetime64("2000-01-01T00:00:00", "ms")
MILLISECONDS_PER_DAY = 86_400_000.0

#
# def model_time_days_to_datetime64(model_time_days: float) -> np.datetime64:
#     milliseconds = int(round(float(model_time_days) * MILLISECONDS_PER_DAY))
#     return TIME_ORIGIN + np.timedelta64(milliseconds, "ms")
#
#
# def datetime64_to_model_time_days(date_time: np.datetime64) -> float:
#     delta_ms = (np.datetime64(date_time, "ms") - TIME_ORIGIN) / np.timedelta64(1, "ms")
#     return float(delta_ms) / MILLISECONDS_PER_DAY
#

class Model3DBackendMock:
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.locations_1d = locations_1d
        self._heads = np.zeros(len(locations_1d), dtype=float)

    def build_cell_assignment(self) -> None:
        return None

    def initial_heads_to_1d(self) -> List[float]:
        return {site_id: self._heads[i] for i, site_id in enumerate(self.locations_1d)}

    def choose_dt(self, current_time: float, t_end: float) -> float:
        remaining = t_end - current_time
        max_step = np.timedelta64(24*3600, 's')  * 5
        return max(min(max_step, remaining), np.timedelta64(1, 's'))

    def model_step(self, dt: float, contributions) -> np.ndarray:
        _ = dt
        self._heads = np.asarray(contributions, dtype=float)
        return self._heads.copy()

class Model3D:
    def __init__(self, composed:ComposedData, model_3d_cfg: dict, locations_1d):
        self.composed = composed
        self.locations_1d = locations_1d
        backend_class = resolve_named_class(
            model_3d_cfg['backend_class_name'],
            (Model3DBackendMock, Model3DBackend),
        )
        self.backend = backend_class(composed, model_3d_cfg=model_3d_cfg, locations_1d=locations_1d)


    def run_loop(
        self,
        queue_names_out_to_1d: list[str],
        queue_name_in_from_1d: str,
    ):
        start_t = self.composed.start
        time = start_t
        end_t = self.composed.end

        q_3d_to_1d = [Queue(name) for name in queue_names_out_to_1d]
        q_1d_to_3d = Queue(queue_name_in_from_1d)

        self.backend.build_cell_assignment()
        heads_to_1d = self.backend.initial_heads_to_1d()

        while time < end_t:
            dt = self.backend.choose_dt(time, end_t)
            assert  dt > np.timedelta64(0, 's'), f"Non-positive time step: {dt}"

            target_time = time + dt
            LOG.info("[3D] === Step: t=%s -> t=%s ===", time, target_time)

            for i, site_id in enumerate(self.locations_1d):
                head = heads_to_1d[site_id]
                msg_out = Data3DTo1D(
                    date_time=target_time,
                    site_id=site_id,
                    pressure_head=head,
                )
                q_3d_to_1d[i].put(msg_out)
                LOG.info("[3D] send head -> 1D %s: date_time=%s, head=%s", i, msg_out.date_time, head)

            contributions = {}
            while len(contributions) < len(self.locations_1d):
                msg_in = q_1d_to_3d.get()
                assert isinstance(msg_in, Data1DTo3D), f"Unexpected 1D->3D payload: {type(msg_in)}"
                id = int(msg_in.site_id)
                assert id not in contributions, "Duplicate contribution from 1D site_id=%s" % id
                LOG.info("[3D] received from 1D %s: date_time=%s, recharge=%s", idx, msg_in.date_time, msg_in.velocity)
                contributions[id] = float(msg_in.velocity)

            heads_to_1d = self.backend.model_step(dt, contributions)

            time = target_time

        LOG.info(f"[3D] finished time loop at t={time} (t_end={end_t})")
