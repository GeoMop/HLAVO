from __future__ import annotations
from typing import *

import logging
from pathlib import Path

import flopy
from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)

import numpy as np
from dask.distributed import Queue
from modflowapi import ModflowApi

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
import hlavo.deep_model.model_3d_cfg as cfg3d
from hlavo.deep_model.coupled_runtime import Model3DBackend
from hlavo.composed.prediction_writer import prediction_writer_from_config
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

    def well_prediction(self, wells_dataset):
        _ = wells_dataset
        return {}


class Model3DDelay(Model3DBackendMock):
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        super().__init__(composed, model_3d_cfg, locations_1d)
        self.water_level = float(model_3d_cfg["initial_water_level"])

    def model_step(self, dt: float, contributions) -> dict[int, float]:
        dt_days = float(dt / np.timedelta64(1, "D"))
        recharge = np.array([float(contributions[site_id]) for site_id in self.locations_1d], dtype=float)
        self.water_level = self.water_level + float(np.sum(recharge)) * dt_days
        self.water_level = self.water_level - max(self.water_level - (-60.0), 0.0) * dt_days * 0.1
        return {site_id: self.water_level for site_id in self.locations_1d}

    def well_prediction(self, wells_dataset):
        if wells_dataset is None:
            return {}
        return {str(well_id): self.water_level for well_id in wells_dataset["well_id"].values}


class ModflowApiCubeBuilder:
    def __init__(self, model_3d_cfg: dict) -> None:
        self.sim_name = str(model_3d_cfg["sim_name"])
        self.model_name = str(model_3d_cfg["model_name"])
        self.recharge_package = str(model_3d_cfg["recharge_package"])
        self.simulation_days = float(model_3d_cfg["simulation_days"])
        self.time_step_days = float(model_3d_cfg["time_step_hours"]) / 24.0
        self.initial_head = float(model_3d_cfg["initial_head"])
        self.top = float(model_3d_cfg["top"])
        self.bottom = float(model_3d_cfg["bottom"])

    def build(self, workdir: Path) -> None:
        n_steps = int(round(self.simulation_days / self.time_step_days))
        assert n_steps > 0, "MODFLOW API cube must have at least one time step"

        sim = flopy.mf6.MFSimulation(sim_name=self.sim_name, sim_ws=workdir, exe_name="mf6")
        flopy.mf6.ModflowTdis(
            sim,
            nper=1,
            perioddata=[(self.simulation_days, n_steps, 1.0)],
            time_units="DAYS",
        )
        flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
        gwf = flopy.mf6.ModflowGwf(sim, modelname=self.model_name, save_flows=True)
        flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1.0,
            delc=1.0,
            top=self.top,
            botm=self.bottom,
        )
        flopy.mf6.ModflowGwfic(gwf, strt=self.initial_head)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.0)
        flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1.0e-5, sy=0.1)
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), self.initial_head]])
        flopy.mf6.ModflowGwfrcha(gwf, pname=self.recharge_package, recharge=0.0)
        flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{self.model_name}.hds",
            saverecord=[("HEAD", "ALL")],
        )
        sim.write_simulation(silent=True)


class Model3DAPI:
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.model_3d_cfg = model_3d_cfg
        self.locations_1d = locations_1d
        self.builder = ModflowApiCubeBuilder(model_3d_cfg)
        self.mf6 = None
        self.head_tag = None
        self.recharge_tag = None

    def build_cell_assignment(self) -> None:
        self.composed.workdir.mkdir(parents=True, exist_ok=True)
        self.builder.build(self.composed.workdir)
        self.mf6 = ModflowApi(
            self.model_3d_cfg["lib_path"],
            working_directory=str(self.composed.workdir),
        )
        self.mf6.initialize()
        self.head_tag = self.mf6.get_var_address("X", self.builder.model_name)
        self.recharge_tag = self.mf6.get_var_address(
            "RECHARGE",
            self.builder.model_name,
            self.builder.recharge_package,
        )

    def choose_dt(self, current_time: np.datetime64["s"], t_end: np.datetime64["s"]) -> np.timedelta64["s"]:
        remaining = t_end - current_time
        time_step_hours = float(self.model_3d_cfg["time_step_hours"])
        max_step = np.timedelta64(int(round(time_step_hours * 3600)), "s")
        return max(min(max_step, remaining), np.timedelta64(1, "s"))

    def initial_heads_to_1d(self) -> dict[int, float]:
        return self._pressure_heads()

    def model_step(self, dt: np.timedelta64, contributions) -> dict[int, float]:
        _ = dt
        assert self.mf6 is not None, "MODFLOW API backend is not initialized"
        recharge_m_per_s = np.array([float(contributions[site_id]) for site_id in self.locations_1d], dtype=float)
        recharge_m_per_day = np.full(1, float(np.mean(recharge_m_per_s)) * 86400.0, dtype=float)
        recharge = self.mf6.get_value(self.recharge_tag)
        assert recharge.size == recharge_m_per_day.size, "Unexpected MODFLOW recharge array size"
        recharge[:] = recharge_m_per_day
        self.mf6.set_value(self.recharge_tag, recharge)
        self.mf6.update()
        return self._pressure_heads()

    def well_prediction(self, wells_dataset):
        _ = wells_dataset
        return {}

    def close(self) -> None:
        if self.mf6 is not None:
            self.mf6.finalize()
            self.mf6 = None

    def _pressure_heads(self) -> dict[int, float]:
        assert self.mf6 is not None, "MODFLOW API backend is not initialized"
        heads = np.asarray(self.mf6.get_value(self.head_tag), dtype=float)
        pressure_head = float(heads.reshape((1, 1, 1))[0, 0, 0] - self.builder.top)
        return {site_id: pressure_head for site_id in self.locations_1d}


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
        self.writer = prediction_writer_from_config(composed, locations_1d, common_cfg)


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
            site_messages = []
            while len(contributions) < len(self.locations_1d):
                msg_in = q_1d_to_3d.get()
                assert isinstance(msg_in, Data1DTo3D), f"Unexpected 1D->3D payload: {type(msg_in)}"
                id = int(msg_in.site_id)
                assert id not in contributions, "Duplicate contribution from 1D site_id=%s" % id
                LOG.info("[3D] received from 1D %s: date_time=%s, recharge=%s", id, msg_in.date_time, msg_in.velocity)
                contributions[id] = float(msg_in.velocity)
                site_messages.append(msg_in)

            heads_to_1d = self.backend.model_step(dt, contributions)
            if self.writer is not None:
                site_messages = sorted(site_messages, key=lambda msg: int(msg.site_id))
                well_prediction = self.backend.well_prediction(self.writer.wells)
                self.writer.write_step(target_time, site_messages, heads_to_1d, well_prediction)

            time = target_time

        LOG.info(f"[3D] finished time loop at t={time} (t_end={end_t})")
        if self.writer is not None:
            self.writer.close()
        if hasattr(self.backend, "close"):
            self.backend.close()
        return heads_to_1d
