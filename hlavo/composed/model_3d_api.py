"""MODFLOW 6 3D backend driven through the MODFLOW API (libmf6 via modflowapi).

Unlike ``deep_model.coupled_runtime.Model3DBackend`` (which rewrites the input
files and restarts ``mf6`` for every coupling step), the simulation is built
and initialized once and then advanced one MODFLOW time step per composed
step from our own time loop in ``Model3D.run_loop``:

    (1D models run) -> prescribe recharge -> solve the MF6 time step
    -> read water-table heads -> return pressure heads for the 1D models

Geometry, materials and boundary conditions are *not* defined here. They come
from a pluggable model builder (``model_builder: "module:function"`` in the
config) so tests can inject a toy geometry now and the GIS based geometry
(milestones 4-5) can plug in later. The backend owns time discretization,
the solver and the coupling.

Units: MODFLOW runs in days and meters; the 1D recharge (Darcy velocity) is
taken in m/day, pressure heads returned to 1D are ``head - top`` in meters
(negative below the model top), consistent with ``Model3DDelay``.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Callable

import attrs
import flopy
import numpy as np
from modflowapi import ModflowApi

from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)

DRY_HEAD_ABS_THRESHOLD = 1.0e20
NO_SITE = -1
SECONDS_PER_DAY = 86400.0


@attrs.define(frozen=True)
class GwfSetup:
    """What a model builder hands back to the backend.

    gwf:              flopy groundwater-flow model added to the given simulation.
    recharge_package: package name of the array-based recharge (RCHA) package
                      the coupling writes to; recharge acts on the top layer.
    cell_owner:       (nrow, ncol) array of 1D site_id owning each column, NO_SITE for none.
    """
    gwf: flopy.mf6.ModflowGwf
    recharge_package: str
    cell_owner: np.ndarray


ModelBuilder = Callable[[flopy.mf6.MFSimulation, list, dict], GwfSetup]


@attrs.define(frozen=True)
class GridMap:
    """Structured-grid bookkeeping needed to move data between MF6 and the sites."""
    shape: tuple[int, int, int]           # (nlay, nrow, ncol)
    top: np.ndarray                       # (nrow, ncol) model top elevation
    node_user: np.ndarray                 # 0-based user node for each reduced MF6 node
    site_columns: dict[int, np.ndarray]   # site_id -> flat (row*ncol+col) column indices

    @classmethod
    def from_setup(cls, setup: GwfSetup, node_user: np.ndarray, locations_1d: list[int]) -> "GridMap":
        dis = setup.gwf.dis
        shape = (int(dis.nlay.array), int(dis.nrow.array), int(dis.ncol.array))
        top = np.broadcast_to(np.asarray(dis.top.array, dtype=float), shape[1:]).copy()
        owner = np.asarray(setup.cell_owner)
        assert owner.shape == shape[1:], f"cell_owner shape {owner.shape} != (nrow, ncol) {shape[1:]}"
        site_columns = {int(s): np.flatnonzero(owner.ravel() == int(s)) for s in locations_1d}
        empty = [s for s, cols in site_columns.items() if cols.size == 0]
        assert not empty, f"No model columns assigned to 1D sites: {empty}"
        return cls(shape=shape, top=top, node_user=node_user, site_columns=site_columns)

    def full_heads(self, x_reduced: np.ndarray) -> np.ndarray:
        """Heads on the full (nlay, nrow, ncol) grid, NaN for inactive and dry cells."""
        full = np.full(int(np.prod(self.shape)), np.nan)
        full[self.node_user] = x_reduced
        full[np.abs(full) >= DRY_HEAD_ABS_THRESHOLD] = np.nan
        return full.reshape(self.shape)

    def water_table(self, heads: np.ndarray) -> np.ndarray:
        """Head of the uppermost wet cell in every column, flattened to (nrow*ncol,)."""
        wet = np.isfinite(heads)
        first_wet = np.argmax(wet, axis=0)
        table = np.take_along_axis(heads, first_wet[None], axis=0)[0]
        table[~wet.any(axis=0)] = np.nan
        return table.ravel()

    def site_pressure_heads(self, heads: np.ndarray) -> dict[int, float]:
        pressure = self.water_table(heads) - self.top.ravel()
        result = {}
        for site_id, cols in self.site_columns.items():
            values = pressure[cols]
            assert np.isfinite(values).any(), f"All model columns dry for 1D site_id={site_id}"
            result[site_id] = float(np.nanmean(values))
        return result

    def spread_recharge(self, contributions: dict[int, float]) -> np.ndarray:
        """Recharge per top-layer column (m/day); columns without a site get zero."""
        recharge = np.zeros(self.shape[1] * self.shape[2])
        for site_id, cols in self.site_columns.items():
            recharge[cols] = float(contributions[site_id])
        return recharge


def resolve_builder(spec: str) -> ModelBuilder:
    module_name, sep, attr = spec.partition(":")
    assert sep and module_name and attr, f"model_builder must be 'module:function', got {spec!r}"
    return getattr(importlib.import_module(module_name), attr)


def default_libmf6() -> Path:
    suffix = {"darwin": "libmf6.dylib", "win32": "libmf6.dll"}.get(sys.platform, "libmf6.so")
    for prefix in (sys.prefix, sys.base_prefix):
        for sub in ("lib", "Library/bin"):
            candidate = Path(prefix) / sub / suffix
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"{suffix} not found under {sys.prefix} or {sys.base_prefix}; set model_3d.common.lib_path")


class Model3DAPI:
    """3D backend running MODFLOW 6 in-process through the MODFLOW API.

    Config (``model_3d.common``):
      time_step_hours:  coupling step, one MF6 time step each (default 24)
      model_builder:    "module:function" building the GWF model, see ``ModelBuilder``
      model_builder_config: mapping passed to the builder (optional)
      lib_path:         path to libmf6 (optional, found in the Python env by default)
      ims:              flopy ModflowIms keyword overrides (optional)
    """

    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.cfg = model_3d_cfg
        self.locations_1d = [int(s) for s in locations_1d]
        hours = float(model_3d_cfg.get("time_step_hours", 24.0))
        self.time_step = np.timedelta64(int(round(hours * 3600)), "s")
        self.sim_ws = Path(composed.workdir) / "model_3d"
        self.mf6: ModflowApi | None = None
        self.grid: GridMap | None = None
        self._model = self._recharge_pkg = None
        self._head_addr = None

    # ---- setup -----------------------------------------------------------
    def _time_discretization(self) -> tuple[float, int]:
        total = self.composed.end - self.composed.start
        n_steps, rest = divmod(total, self.time_step)
        assert n_steps > 0 and rest == np.timedelta64(0, "s"), (
            f"Simulated interval {total} must be a positive multiple of time_step {self.time_step}"
        )
        return float(total / np.timedelta64(1, "s")) / SECONDS_PER_DAY, int(n_steps)

    def _build_simulation(self) -> GwfSetup:
        self.sim_ws.mkdir(parents=True, exist_ok=True)
        sim = flopy.mf6.MFSimulation(sim_name="model_3d", sim_ws=str(self.sim_ws), exe_name="mf6")
        perlen, nstp = self._time_discretization()
        flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)])
        builder = resolve_builder(self.cfg["model_builder"])
        setup = builder(sim, self.locations_1d, dict(self.cfg.get("model_builder_config", {})))
        assert isinstance(setup, GwfSetup), f"model builder must return GwfSetup, got {type(setup)}"
        ims_kwargs = {"complexity": "MODERATE", **self.cfg.get("ims", {})}
        ims = flopy.mf6.ModflowIms(sim, **ims_kwargs)
        sim.register_ims_package(ims, [setup.gwf.name])
        sim.write_simulation(silent=True)
        return setup

    def initialize(self) -> None:
        """Build and write the simulation, load libmf6 and initialize it (called once)."""
        if self.mf6 is not None:
            return
        setup = self._build_simulation()
        lib_path = Path(self.cfg["lib_path"]) if "lib_path" in self.cfg else default_libmf6()
        LOG.info("[3D API] initializing MODFLOW 6 (%s) in %s", lib_path, self.sim_ws)
        self.mf6 = ModflowApi(str(lib_path), working_directory=str(self.sim_ws))
        self.mf6.initialize()

        self._model = setup.gwf.name.upper()
        self._recharge_pkg = setup.recharge_package.upper()
        self._head_addr = self.mf6.get_var_address("X", self._model)
        node_user = np.asarray(self._get("NODEUSER", "DIS"), dtype=int)
        n_reduced = self.mf6.get_value_ptr(self._head_addr).size
        # MF6 allocates NODEUSER with a single entry when IDOMAIN removes no cells.
        node_user = np.arange(n_reduced) if node_user.size != n_reduced else node_user - 1
        self.grid = GridMap.from_setup(setup, node_user, self.locations_1d)

    def _get(self, var: str, component: str | None = None) -> np.ndarray:
        address = self.mf6.get_var_address(var, self._model, *([component] if component else []))
        return self.mf6.get_value(address)

    def _set_recharge(self, recharge_columns: np.ndarray) -> None:
        """Write recharge (m/day per top-layer column) into the RCH package.

        The RCH solver term uses the package's RECHARGE array (``rch_cf``: rhs = -recharge * area),
        entry i applied to cell NODELIST(i) (reduced, 1-based). Do NOT write BOUND: for RCH it is
        allocated with zero columns and xmipy drops zero dimensions from the shape, so a pointer
        to it overruns the allocation (heap corruption). The value set after prepare_time_step
        is kept for the step since a single stress period never re-reads the input.
        """
        nbound = int(np.ravel(self._get("NBOUND", self._recharge_pkg))[0])
        nodes = np.asarray(self._get("NODELIST", self._recharge_pkg), dtype=int)[:nbound] - 1
        ncpl = self.grid.shape[1] * self.grid.shape[2]
        columns = self.grid.node_user[nodes] % ncpl
        address = self.mf6.get_var_address("RECHARGE", self._model, self._recharge_pkg)
        recharge = self.mf6.get_value_ptr(address)
        assert recharge.ndim == 1 and recharge.size >= nbound, f"Unexpected RECHARGE shape {recharge.shape}"
        recharge[:nbound] = recharge_columns[columns]

    # Model3D.run_loop entry point, kept under the name used by the other backends.
    def build_cell_assignment(self) -> None:
        self.initialize()

    # ---- coupling --------------------------------------------------------
    def _pressure_heads(self) -> dict[int, float]:
        heads = self.grid.full_heads(self.mf6.get_value_ptr(self._head_addr))
        return self.grid.site_pressure_heads(heads)

    def initial_heads_to_1d(self) -> dict[int, float]:
        self.initialize()
        return self._pressure_heads()

    def choose_dt(self, current_time: np.datetime64, t_end: np.datetime64) -> np.timedelta64:
        remaining = t_end - current_time
        assert remaining >= self.time_step, f"Remaining {remaining} shorter than MF6 time step {self.time_step}"
        return self.time_step

    def model_step(self, dt: np.timedelta64, contributions) -> dict[int, float]:
        assert self.mf6 is not None, "Model3DAPI.initialize() was not called"
        dt_days = float(dt / np.timedelta64(1, "s")) / SECONDS_PER_DAY
        self.mf6.prepare_time_step(dt_days)
        mf6_dt = self.mf6.get_time_step()
        assert abs(mf6_dt - dt_days) < 1e-9 * max(1.0, dt_days), f"MF6 step {mf6_dt} d != coupling step {dt_days} d"
        self._set_recharge(self.grid.spread_recharge(contributions))
        self.mf6.do_time_step()
        self.mf6.finalize_time_step()
        return self._pressure_heads()

    def well_prediction(self, wells_dataset):
        _ = wells_dataset  # monitoring wells: milestone 7
        return {}

    def close(self) -> None:
        if self.mf6 is not None:
            self.mf6.finalize()
            self.mf6 = None
