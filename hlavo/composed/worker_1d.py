from __future__ import annotations

import logging

import numpy as np
from dask.distributed import Queue

from hlavo.composed.common_data import ComposedData
from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.kalman.model_1d import Model1D

LOG = logging.getLogger(__name__)


class Worker1D:
    def __init__(self, composed: ComposedData, site_id: int, config: dict):
        self.composed = composed
        self.site_id = site_id
        self.model = Model1D.from_config(
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


def model1d_worker_entry(composed: ComposedData, site_idx, location, config, queue_name_in, queue_name_out):
    model = Worker1D(
        composed=composed,
        site_id=site_idx,
        location=location,
        config=config,
    )
    return model.run_loop(queue_name_in, queue_name_out)
