from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np

from hlavo.misc.config import (
    get_path,
    load_config,
    optional_mapping,
    require_mapping,
    to_float,
    to_optional_float,
    to_optional_int,
)

# AGENT: Do not put all config datacalsses into single source
# only put here the common config dataclass and its initialization from given dict
# Try to use reusable validator functions instead of a single from_dict staticmethod
# where validation/conversion of individual attributes is obfuscated by boilerplate
# End it is hard to maintain consistent checks and log messages of simmilar checks.
#
# In particular keep geometry config in qgis_reader just rename MoedlConfig to GeometryConfig and possibly
# add some attribute.
# Keep material config in add_material_parameters.py
# In build_modflow_grid.py clearly separate buidling the geometry (using qgis reader) and setup materials.
# You can deconstruct BuildConfig, partly covered by Model3DCommonConfig,
# Minimize changes (with respect to commited version) while performing requered separation.

MODEL_3D_STEP_DAYS = 5.0
MODEL_DIRNAME = "model_with_mine"
GRID_MATERIALS_FILENAME = "grid_materials.npz"
MATERIAL_PARAMETERS_FILENAME = "material_parameters.npz"
PARAVIEW_RESULTS_FILENAME = "uhelna_results.vtu"
PARAVIEW_MATERIALS_FILENAME = "uhelna_materials.vtr"
PLOTS_DIRNAME = "plots"


def resolve_model_3d_section(raw: dict) -> dict:
    if "model_3d" in raw:
        model_3d = raw["model_3d"]
        assert isinstance(model_3d, dict), "model_3d config must be a mapping"
        return model_3d
    return raw


def resolve_model_3d_common_raw(raw: dict) -> dict:
    model_3d = resolve_model_3d_section(raw)
    if "common" in model_3d:
        common_raw = model_3d["common"]
        assert isinstance(common_raw, dict), "model_3d.common must be a mapping"
        return common_raw
    model_raw = raw.get("model")
    if model_raw is not None:
        assert isinstance(model_raw, dict), "model config must be a mapping"
        return model_raw
    return model_3d


def resolve_workspace_root(workspace_override: Path | None, common_raw: dict) -> Path:
    if workspace_override is not None:
        return Path(workspace_override).resolve()
    return Path(str(common_raw.get("workspace", "model"))).resolve()


def resolve_model_workspace(workspace_root: Path, common: "Model3DCommonConfig") -> Path:
    return workspace_root / common.model_name


def resolve_model_relative_path(workspace: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return workspace / path


@attrs.define(frozen=True)
class Model3DCommonConfig:
    backend_class_name: str
    model_name: str
    sim_name: str
    exe_name: str
    recharge_rate: float
    recharge_series_m_per_day: tuple[float, ...] | None
    drain_conductance: float
    total_time_days: float | None
    n_steps: int | None
    simulation_days: float
    stress_periods_days: tuple[float, ...]

    @classmethod
    def from_mapping(cls, raw: dict) -> "Model3DCommonConfig":
        assert isinstance(raw, dict), "3D common config must be a mapping"
        model_name = MODEL_DIRNAME
        sim_name = str(raw.get("sim_name", raw.get("name", "uhelna")))
        exe_name = str(raw.get("exe_name", raw.get("executable", "mf6")))
        backend_class_name = str(raw.get("backend_class_name", raw.get("class_name", "Model3DBackend")))
        recharge_rate = to_float(raw, "recharge_rate", 1.0e-4)

        recharge_series_raw = raw.get("recharge_series_m_per_day")
        if recharge_series_raw is None:
            recharge_series_m_per_day = None
        else:
            assert isinstance(recharge_series_raw, (list, tuple)), (
                "recharge_series_m_per_day must be a list"
            )
            recharge_series_m_per_day = tuple(float(value) for value in recharge_series_raw)
            assert all(np.isfinite(value) for value in recharge_series_m_per_day), (
                "recharge_series_m_per_day must contain finite values"
            )

        drain_conductance = to_float(raw, "drain_conductance", 1.0)
        total_time_days = to_optional_float(raw, "total_time_days")
        n_steps = to_optional_int(raw, "n_steps")

        simulation_days_default = (
            total_time_days
            if total_time_days is not None
            else MODEL_3D_STEP_DAYS * float(n_steps) if n_steps is not None else 1.0
        )
        simulation_days = to_float(raw, "simulation_days", simulation_days_default)
        assert simulation_days > 0.0, "simulation_days must be > 0"

        stress_periods_raw = raw.get("stress_periods_days")
        if stress_periods_raw is None:
            stress_periods_days = (simulation_days,)
        else:
            assert isinstance(stress_periods_raw, (list, tuple)), "stress_periods_days must be a list"
            stress_periods_days = tuple(float(value) for value in stress_periods_raw)
            assert all(value > 0.0 for value in stress_periods_days), (
                "stress_periods_days values must be > 0"
            )

        total_period_days = float(sum(stress_periods_days))
        assert np.isclose(total_period_days, simulation_days, rtol=1.0e-6, atol=1.0e-6), (
            "Sum of stress_periods_days must equal simulation_days"
        )
        if recharge_series_m_per_day is not None:
            assert len(recharge_series_m_per_day) == len(stress_periods_days), (
                "recharge_series_m_per_day length must match stress periods"
            )

        return cls(
            backend_class_name=backend_class_name,
            model_name=model_name,
            sim_name=sim_name,
            exe_name=exe_name,
            recharge_rate=recharge_rate,
            recharge_series_m_per_day=recharge_series_m_per_day,
            drain_conductance=drain_conductance,
            total_time_days=total_time_days,
            n_steps=n_steps,
            simulation_days=simulation_days,
            stress_periods_days=stress_periods_days,
        )

    @classmethod
    def from_source(cls, config_source: Path | dict) -> tuple["Model3DCommonConfig", Path | None, dict]:
        raw, config_path = load_config(config_source)
        common_raw = resolve_model_3d_common_raw(raw)
        return cls.from_mapping(common_raw), config_path, raw

    def resolve_t_end(self) -> float:
        if self.total_time_days is not None:
            t_end = self.total_time_days
        elif self.n_steps is not None:
            t_end = MODEL_3D_STEP_DAYS * float(self.n_steps)
        else:
            t_end = MODEL_3D_STEP_DAYS
        assert t_end > 0.0, "model_3d total simulation time must be > 0"
        return float(t_end)
