from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import attrs
import flopy
import numpy as np
from modflowapi import ModflowApi

from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)


def _time_step(step_hours: float) -> np.timedelta64:
    return np.timedelta64(int(round(float(step_hours) * 3600.0)), "s")


def _step_schedule(
    start_time: np.datetime64,
    end_time: np.datetime64,
    *,
    step_hours: float,
) -> tuple[np.timedelta64, ...]:
    dt = _time_step(step_hours)
    assert dt > np.timedelta64(0, "s"), "time_step_hours must be positive"
    steps: list[np.timedelta64] = []
    current_time = np.datetime64(start_time, "s")
    end_time = np.datetime64(end_time, "s")
    while current_time < end_time:
        step = min(dt, end_time - current_time)
        steps.append(step)
        current_time += step
    assert steps, "At least one MODFLOW API time step is required"
    return tuple(steps)


@attrs.define(frozen=True)
class ApiBuildResult:
    sim_name: str
    recharge_package_name: str
    site_index_by_id: dict[int, int]


@attrs.define
class SimpleCubeModelBuilder:
    sim_name: str
    top: float
    bottom: float
    initial_head: float
    hydraulic_conductivity: float
    specific_storage: float
    specific_yield: float
    delr: float
    delc: float

    @classmethod
    def from_config(
        cls,
        *,
        model_3d_cfg: dict,
        locations_1d: list[int],
    ) -> "SimpleCubeModelBuilder":
        builder_cfg = model_3d_cfg.get("builder", {})
        assert isinstance(builder_cfg, dict), "model_3d.common.builder must be a mapping"
        _ = locations_1d
        return cls(
            sim_name=str(builder_cfg.get("sim_name", "cube")),
            top=float(builder_cfg.get("top", 0.0)),
            bottom=float(builder_cfg.get("bottom", -10.0)),
            initial_head=float(builder_cfg.get("initial_head", 0.0)),
            hydraulic_conductivity=float(builder_cfg.get("hydraulic_conductivity", 1.0)),
            specific_storage=float(builder_cfg.get("specific_storage", 1.0e-5)),
            specific_yield=float(builder_cfg.get("specific_yield", 0.1)),
            delr=float(builder_cfg.get("delr", 1.0)),
            delc=float(builder_cfg.get("delc", 1.0)),
        )

    def build(
        self,
        *,
        workspace: Path,
        step_schedule: tuple[np.timedelta64, ...],
        locations_1d: list[int],
        exe_name: str,
    ) -> ApiBuildResult:
        n_sites = len(locations_1d)
        assert n_sites > 0, "At least one 1D site is required for the API cube builder"
        perioddata = [(float(step / np.timedelta64(1, "D")), 1, 1.0) for step in step_schedule]
        sim = flopy.mf6.MFSimulation(sim_name=self.sim_name, exe_name=exe_name, sim_ws=str(workspace))
        flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(perioddata), perioddata=perioddata)
        flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

        gwf = flopy.mf6.ModflowGwf(sim, modelname=self.sim_name, save_flows=True)
        flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=1,
            nrow=1,
            ncol=n_sites,
            delr=self.delr,
            delc=self.delc,
            top=self.top,
            botm=[self.bottom],
            idomain=1,
        )
        flopy.mf6.ModflowGwfic(
            gwf,
            strt=np.full((1, 1, n_sites), self.initial_head, dtype=float),
        )
        flopy.mf6.ModflowGwfnpf(
            gwf,
            icelltype=[1],
            k=self.hydraulic_conductivity,
            k33=self.hydraulic_conductivity,
            save_specific_discharge=True,
        )
        flopy.mf6.ModflowGwfsto(
            gwf,
            iconvert=[1],
            ss=self.specific_storage,
            sy=self.specific_yield,
            transient={iper: True for iper in range(len(perioddata))},
        )
        zero_recharge = np.zeros((1, n_sites), dtype=float)
        flopy.mf6.ModflowGwfrcha(
            gwf,
            recharge={iper: zero_recharge for iper in range(len(perioddata))},
            pname="RCH-1",
        )
        flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{self.sim_name}.hds",
            budget_filerecord=f"{self.sim_name}.cbc",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )
        sim.write_simulation()
        site_index_by_id = {int(site_id): idx for idx, site_id in enumerate(locations_1d)}
        return ApiBuildResult(
            sim_name=self.sim_name,
            recharge_package_name="RCH-1",
            site_index_by_id=site_index_by_id,
        )


def _resolve_builder_class(builder_class_name: str) -> type[SimpleCubeModelBuilder]:
    assert builder_class_name == "SimpleCubeModelBuilder", (
        "Only SimpleCubeModelBuilder is currently supported for Model3DAPI tests"
    )
    return SimpleCubeModelBuilder


def _resolve_lib_path(lib_path: str | None, exe_name: str) -> Path:
    if lib_path is not None:
        path = Path(lib_path)
        assert path.exists(), f"Configured modflowapi library does not exist: {path}"
        return path
    env_lib_path = os.environ.get("LIBMF6_PATH")
    if env_lib_path:
        path = Path(env_lib_path)
        assert path.exists(), f"Configured LIBMF6_PATH does not exist: {path}"
        return path

    python_prefix = Path(sys.executable).resolve().parent.parent
    candidate_paths = [
        python_prefix / "lib" / "libmf6.so",
    ]
    executable = shutil.which(exe_name)
    if executable is not None:
        executable_path = Path(executable).resolve()
        candidate_paths.append(executable_path.parent.parent / "lib" / "libmf6.so")

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    assert False, (
        f"Unable to locate libmf6.so; checked LIBMF6_PATH, "
        f"{python_prefix / 'lib' / 'libmf6.so'}, and executable '{exe_name}'"
    )


class Model3DAPI:
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.locations_1d = [int(site_id) for site_id in locations_1d]
        self.model_3d_cfg = model_3d_cfg
        self.step_schedule = _step_schedule(
            composed.start,
            composed.end,
            step_hours=float(model_3d_cfg["time_step_hours"]),
        )
        builder_class = _resolve_builder_class(model_3d_cfg.get("builder_class_name", "SimpleCubeModelBuilder"))
        self.builder = builder_class.from_config(model_3d_cfg=model_3d_cfg, locations_1d=self.locations_1d)
        self.build_result = self.builder.build(
            workspace=composed.workdir,
            step_schedule=self.step_schedule,
            locations_1d=self.locations_1d,
            exe_name=str(model_3d_cfg.get("exe_name", "mf6")),
        )
        lib_path = _resolve_lib_path(model_3d_cfg.get("lib_path"), str(model_3d_cfg.get("exe_name", "mf6")))
        self.api = ModflowApi(str(lib_path), working_directory=str(composed.workdir))
        self.api.initialize()
        self.recharge_ptr = self.api.get_value_ptr(
            self.api.get_var_address(
                "RECHARGE",
                self.build_result.sim_name,
                self.build_result.recharge_package_name,
            )
        )
        self.head_ptr = self.api.get_value_ptr(
            self.api.get_var_address("X", self.build_result.sim_name)
        )
        self._step_index = 0
        self._well_ids: tuple[str, ...] = ()
        self._current_heads = self._heads_to_1d()
        LOG.info(
            "[3D API] initialized modflowapi sim=%s lib=%s steps=%s",
            self.build_result.sim_name,
            lib_path,
            len(self.step_schedule),
        )

    def build_cell_assignment(self) -> None:
        return None

    def set_well_metadata(self, well_metadata) -> None:
        self._well_ids = tuple(item.well_id for item in well_metadata)

    def initial_heads_to_1d(self) -> dict[int, float]:
        return dict(self._current_heads)

    def choose_dt(self, current_time: np.datetime64, t_end: np.datetime64) -> np.timedelta64:
        _ = current_time
        _ = t_end
        return self.step_schedule[self._step_index]

    def _site_recharge_array(self, contributions: dict[int, float]) -> np.ndarray:
        recharge = np.zeros_like(self.recharge_ptr, dtype=float)
        for site_id, value in contributions.items():
            recharge[self.build_result.site_index_by_id[int(site_id)]] = float(value)
        return recharge

    def _heads_to_1d(self) -> dict[int, float]:
        return {
            site_id: float(self.head_ptr[index])
            for site_id, index in self.build_result.site_index_by_id.items()
        }

    def model_step(self, dt: np.timedelta64, contributions) -> dict[int, float]:
        expected_dt = self.step_schedule[self._step_index]
        assert dt == expected_dt, f"Unexpected API dt {dt}, expected {expected_dt}"
        self.recharge_ptr[:] = self._site_recharge_array(contributions)
        self.api.update()
        self._step_index += 1
        self._current_heads = self._heads_to_1d()
        return dict(self._current_heads)

    def read_well_levels(self) -> dict[str, float]:
        if not self._well_ids:
            return {}
        mean_head = float(np.mean(list(self._current_heads.values())))
        return {well_id: mean_head for well_id in self._well_ids}

    def finalize(self) -> None:
        self.api.finalize()
