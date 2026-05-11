from __future__ import annotations
from typing import *

import logging

from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)

import numpy as np
from dask.distributed import Queue

from hlavo.composed.model_3d_api import Model3DAPI
from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.composed.writer import JsonlModel3DWriter, NullModel3DWriter, ZarrModel3DWriter
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
        self.model_3d_cfg = model_3d_cfg
        self._heads = np.zeros(len(locations_1d), dtype=float)

    def build_cell_assignment(self) -> None:
        return None

    def initial_heads_to_1d(self) -> List[float]:
        return {site_id: self._heads[i] for i, site_id in enumerate(self.locations_1d)}

    def choose_dt(self, current_time: float, t_end: float) -> float:
        remaining = t_end - current_time
        time_step_hours = float(self.model_3d_cfg["time_step_hours"])
        max_step = np.timedelta64(int(round(time_step_hours * 3600)), "s")
        return max(min(max_step, remaining), np.timedelta64(1, 's'))

    def model_step(self, dt: float, contributions) -> np.ndarray:
        dt_days = float(dt / np.timedelta64(1, "D"))
        recharge = np.array([float(contributions[site_id]) for site_id in self.locations_1d], dtype=float)
        self._heads = self._heads + recharge * dt_days
        return {site_id: self._heads[i] for i, site_id in enumerate(self.locations_1d)}


class Model3DDelay:
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.locations_1d = locations_1d
        self.model_3d_cfg = model_3d_cfg
        self.reference_water_level = float(model_3d_cfg.get("reference_water_level", -60.0))
        self.water_level = self.reference_water_level
        self._well_ids: tuple[str, ...] = ()

    def build_cell_assignment(self) -> None:
        return None

    def set_well_metadata(self, well_metadata) -> None:
        self._well_ids = tuple(item.well_id for item in well_metadata)

    def initial_heads_to_1d(self) -> List[float]:
        return {site_id: self.water_level for site_id in self.locations_1d}

    def choose_dt(self, current_time: float, t_end: float) -> float:
        remaining = t_end - current_time
        time_step_hours = float(self.model_3d_cfg["time_step_hours"])
        max_step = np.timedelta64(int(round(time_step_hours * 3600)), "s")
        return max(min(max_step, remaining), np.timedelta64(1, "s"))

    def model_step(self, dt: float, contributions) -> np.ndarray:
        dt_days = float(dt / np.timedelta64(1, "D"))
        self.water_level += float(sum(contributions.values())) * dt_days
        positive_part = max(self.water_level - self.reference_water_level, 0.0)
        self.water_level -= positive_part * dt_days * 0.1
        return {site_id: self.water_level for site_id in self.locations_1d}

    def read_well_levels(self) -> dict[str, float]:
        return {well_id: self.water_level for well_id in self._well_ids}

class Model3D:
    def __init__(self, composed:ComposedData, model_3d_cfg: dict, locations_1d):
        self.composed = composed
        self.locations_1d = locations_1d
        common_cfg = model_3d_cfg["common"]
        backend_class = resolve_named_class(
            common_cfg['backend_class_name'],
            (Model3DBackendMock, Model3DDelay, Model3DAPI, Model3DBackend),
        )
        self.backend = backend_class(composed, model_3d_cfg=common_cfg, locations_1d=locations_1d)
        writer_class = resolve_named_class(
            common_cfg.get("writer_class_name", "NullModel3DWriter"),
            (NullModel3DWriter, JsonlModel3DWriter, ZarrModel3DWriter),
        )
        self.writer = writer_class.from_config(
            composed=composed,
            locations_1d=locations_1d,
            writer_config=common_cfg.get("writer", {}),
            calibration_time=composed.start,
        )
        if hasattr(self.backend, "set_well_metadata"):
            self.backend.set_well_metadata(self.writer.well_metadata)


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

        try:
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

                incoming = {}
                while len(incoming) < len(self.locations_1d):
                    msg_in = q_1d_to_3d.get()
                    assert isinstance(msg_in, Data1DTo3D), f"Unexpected 1D->3D payload: {type(msg_in)}"
                    id = int(msg_in.site_id)
                    assert id not in incoming, "Duplicate contribution from 1D site_id=%s" % id
                    LOG.info("[3D] received from 1D %s: date_time=%s, recharge=%s", id, msg_in.date_time, msg_in.velocity)
                    incoming[id] = msg_in

                self.writer.write_site_step(
                    target_time,
                    [
                        {
                            "date_time": np.datetime64(target_time, "m"),
                            "site_id": site_id,
                            "longitude": float(incoming[site_id].longitude),
                            "latitude": float(incoming[site_id].latitude),
                            "velocity": float(incoming[site_id].velocity),
                            "pressure_head": float(heads_to_1d[site_id]),
                        }
                        for site_id in self.locations_1d
                    ],
                )

                contributions = {
                    site_id: float(message.velocity)
                    for site_id, message in incoming.items()
                }
                heads_to_1d = self.backend.model_step(dt, contributions)

                if hasattr(self.backend, "read_well_levels"):
                    well_levels = self.backend.read_well_levels()
                    if well_levels:
                        self.writer.write_well_step(target_time, well_levels)

                time = target_time
        finally:
            if hasattr(self.backend, "finalize"):
                self.backend.finalize()
            self.writer.close()

        LOG.info(f"[3D] finished time loop at t={time} (t_end={end_t})")
        return heads_to_1d
