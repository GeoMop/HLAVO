from __future__ import annotations

import logging

import numpy as np
from dask.distributed import Queue

from hlavo.composed.composed_protocol import Data1DTo3D, Data3DTo1D, as_datetime64_ms, as_timedelta64_ms

LOGGER = logging.getLogger(__name__)


class Model3D:
    def __init__(
        self,
        n_1d: int,
        initial_state: float = 0.0,
        initial_time: np.datetime64 = np.datetime64("2020-01-01T00:00:00.000"),
        base_dt: np.timedelta64 = np.timedelta64(60, "m"),
    ):
        self.n_1d = n_1d
        self.state = float(initial_state)
        self.time = as_datetime64_ms(initial_time, "initial_time")
        self.base_dt = as_timedelta64_ms(base_dt, "base_dt")

    def choose_dt(self, t_end: np.datetime64) -> np.timedelta64:
        remaining = t_end - self.time
        if remaining <= np.timedelta64(0, "ms"):
            return np.timedelta64(0, "ms")
        if remaining < self.base_dt:
            return remaining
        return self.base_dt

    def step(self, target_time: np.datetime64, contributions: list[Data1DTo3D]) -> float:
        LOGGER.info(
            "[3D] step to t=%s, current_state=%s, contributions=%s",
            target_time,
            self.state,
            [item.velocity for item in contributions],
        )
        self.state += float(np.sum([item.velocity for item in contributions]))
        self.time = target_time
        LOGGER.info("[3D] new state=%s", self.state)
        return self.state

    def run_loop(self, t_end: np.datetime64, queue_names_out_to_1d: list[str], queue_name_in_from_1d: str) -> float:
        t_end = as_datetime64_ms(t_end, "t_end")
        q_3d_to_1d = [Queue(name) for name in queue_names_out_to_1d]
        q_1d_to_3d = Queue(queue_name_in_from_1d)

        while self.time < t_end:
            dt = self.choose_dt(t_end)
            if dt <= np.timedelta64(0, "ms"):
                LOGGER.warning("[3D] dt <= 0, stopping to avoid infinite loop.")
                break

            target_time = self.time + dt
            LOGGER.info("[3D] === Step: t=%s -> t=%s ===", self.time, target_time)
            LOGGER.info("[3D] current state=%s", self.state)

            for i in range(self.n_1d):
                data_for_i = self.state + i
                payload = Data3DTo1D.build(
                    date_time=target_time,
                    dt=dt,
                    site_id=i,
                    pressure_head=data_for_i,
                )
                LOGGER.info(
                    "[3D] sending to 1D %s: t=%s, pressure_head=%s",
                    i,
                    target_time,
                    data_for_i,
                )
                q_3d_to_1d[i].put(payload)

            contributions: list[Data1DTo3D | None] = [None] * self.n_1d
            received = 0
            while received < self.n_1d:
                contribution = q_1d_to_3d.get()
                if not isinstance(contribution, Data1DTo3D):
                    raise TypeError(f"Expected Data1DTo3D from queue, got {type(contribution)}")
                idx = contribution.site_id
                if idx < 0 or idx >= self.n_1d:
                    raise ValueError(f"Invalid site_id from 1D contribution: {idx}")
                if contributions[idx] is not None:
                    raise ValueError(f"Duplicate contribution for site_id={idx}")
                if contribution.date_time != target_time:
                    raise ValueError(
                        f"Contribution time mismatch for site_id={idx}: {contribution.date_time} != {target_time}"
                    )
                LOGGER.info(
                    "[3D] received from 1D %s: t=%s, velocity=%s",
                    idx,
                    contribution.date_time,
                    contribution.velocity,
                )
                contributions[idx] = contribution
                received += 1

            if any(item is None for item in contributions):
                raise RuntimeError("Missing one or more 1D contributions")
            self.step(target_time, [item for item in contributions if item is not None])

        for queue_3d_to_1d_site in q_3d_to_1d:
            queue_3d_to_1d_site.put(None)

        LOGGER.info("[3D] finished time loop at t=%s (t_end=%s), state=%s", self.time, t_end, self.state)
        return self.state
