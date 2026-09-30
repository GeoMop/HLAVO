"""3D groundwater model: MODFLOW 6 run in-process through the MODFLOW API.

This module is the `Model3DAPI` backend of `Model3D` (see `hlavo/composed/model_3d.py`).
It builds a MODFLOW 6 simulation once, loads the MODFLOW 6 shared library into the Python
process and advances the simulation one time step per coupling step, exchanging data with
the 1D surface models between the steps. The older `deep_model.coupled_runtime.Model3DBackend`
instead rewrites the input files and starts the `mf6` executable anew for every step.


Background: the MODFLOW 6 API
=============================

MODFLOW 6 (MF6) is distributed as the executable `mf6` and as the shared library `libmf6`
(both in the conda package `modflow6`). The two contain the same solver and read the same
input files. The library implements the Basic Model Interface (BMI, a standard API of
simulation codes) extended by MF6's "XMI". XMI lets the caller run a simulation time step by
time step and read or change MF6's internal arrays between the steps. We use it through the
Python package `modflowapi`: `ModflowApi` is a thin subclass of `xmipy.XmiWrapper`, which loads
the library with `ctypes`. During every call xmipy switches the process working directory to
the simulation folder, so the relative file names inside the MF6 input files resolve.
Reference: MF6 sources `srcbmi/mf6xmi.F90` and `src/mf6core.f90` (checked for MF6 6.6.3).

Life cycle as used by this module
---------------------------------
::

    flopy builds the packages, sim.write_simulation()
                                   -> MF6 input files in <workdir>/model_3d
    mf6 = ModflowApi(libmf6_path, working_directory=<workdir>/model_3d)
                                   -> load the shared library
    mf6.initialize()               -> read mfsim.nam and all input files, allocate the
                                      memory, heads X := starting heads (IC package)
    for every coupling step:
        mf6.prepare_time_step(dt)  -> advance MF6 time to the next time step of TDIS and read
                                      the stress-period input when a new period starts.
                                      MF6 IGNORES `dt`: the step length is fixed by TDIS,
                                      hence the check against mf6.get_time_step().
        <write the recharge>       -> values written between prepare and do replace the input
        mf6.do_time_step()         -> nonlinear (outer) iterations with linear solves
        mf6.finalize_time_step()   -> budgets and output files; if the step did not converge
                                      MF6 returns an error and xmipy raises XMIError
        <read the heads X>
    mf6.finalize()                 -> close the output files, free the memory

Accessing MF6 memory
--------------------
Every MF6 variable is stored by MF6's memory manager under an address
``<MODEL>/<PACKAGE>/<VARIABLE>`` (upper case; model-level variables have no package part),
built by ``mf6.get_var_address(variable, model, package)``.

- ``mf6.get_value(address)`` returns a copy of the array.
- ``mf6.get_value_ptr(address)`` returns a numpy array that shares memory with MF6: writing
  into it changes the model directly (used for the recharge), reading it gives the current
  state without copying (used for the heads).

Pitfall: xmipy builds the numpy view from the shape MF6 reports and drops zero-length
dimensions. A pointer to an array with a zero dimension then covers memory that does not
belong to it, and writing there corrupts the heap (crash later, e.g. in finalize). This happens
for the generic ``BOUND`` array of the recharge package, which MF6 allocates with zero columns.

Variables used here (``<RCH>`` is the name of the array-based recharge package):

======================  ====================================================================
``<MODEL>/X``           heads [m] of the active cells, in reduced numbering (below)
``<MODEL>/DIS/NODEUSER`` user node number (1-based) of each reduced node
``<MODEL>/<RCH>/NBOUND`` number of recharge entries in use
``<MODEL>/<RCH>/NODELIST`` cell (reduced, 1-based) each recharge entry acts on
``<MODEL>/<RCH>/RECHARGE`` recharge rate [m/day] of each entry; the solver uses these values
======================  ====================================================================

Cell numbering. The structured grid of shape (nlay, nrow, ncol) has "user" nodes
``n = (lay * nrow + row) * ncol + col`` (0-based here, 1-based in MF6). MF6 removes the cells
with IDOMAIN <= 0 and numbers the rest consecutively, the "reduced" nodes; X and NODELIST use
reduced numbers. NODEUSER maps reduced to user numbers. When no cell is removed, MF6 allocates
NODEUSER with a single entry and reduced == user numbering.

Dry cells. A convertible cell whose head drops below its bottom is dry; MF6 stores the head
value DHDRY = -1e30 for it. This module treats such heads as NaN.


Coupling implemented here
=========================
Per coupling step (``Model3D.run_loop`` -> ``Model3DAPI.model_step``)::

    1D models run -> recharge written to RCH -> MF6 time step solved
    -> water-table heads read -> pressure heads returned for the 1D models

Geometry, materials and boundary conditions are not defined here. They come from a model
builder named in the config (``model_builder: "module:function"``, see `ModelBuilder`), so a
test injects a toy geometry now and the GIS based geometry (milestones 4-5) plugs in later.
This module owns the time discretization (TDIS), the solver settings (IMS) and the coupling.

Units: MF6 runs in days and meters. The 1D recharge (Darcy velocity) is taken in m/day.
Pressure heads returned to 1D are ``head - top`` in meters (negative below the model top),
consistent with `Model3DDelay`.
?? The 1D velocity unit is not settled: simulation_schema.yaml declares m/s (QaR in PLAN.md).
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

# MF6 marks dry cells by the head DHDRY = -1e30; any |head| above this threshold is treated as dry.
DRY_HEAD_ABS_THRESHOLD = 1.0e20
# `GwfSetup.cell_owner` value of a model column that belongs to no 1D site (gets zero recharge).
NO_SITE = -1
SECONDS_PER_DAY = 86400.0


@attrs.define(frozen=True)
class GwfSetup:
    """Result of a model builder: the groundwater-flow model and how it maps to the 1D sites.

    gwf:              flopy groundwater-flow (GWF) model the builder added to the simulation.
                      It must contain a structured grid (DIS) and an array-based recharge
                      package (``flopy.mf6.ModflowGwfrcha``, "RCHA").
    recharge_package: package name (flopy ``pname``) of that recharge package; the coupling
                      writes the 1D recharge into it. Array-based recharge acts on the top layer,
                      or on the highest active cell of each column.
    cell_owner:       (nrow, ncol) integer array: the 1D site_id owning each model column,
                      `NO_SITE` for columns without a site.
    """
    gwf: flopy.mf6.ModflowGwf
    recharge_package: str
    cell_owner: np.ndarray


# A model builder adds the GWF model to the given simulation (which already contains TDIS)
# and returns its `GwfSetup`. Arguments: (simulation, site_ids of the 1D models,
# `model_builder_config` mapping from the config). Example: tests/composed/modflow_cube.py.
ModelBuilder = Callable[[flopy.mf6.MFSimulation, list, dict], GwfSetup]


@attrs.define(frozen=True)
class GridMap:
    """Bookkeeping to move data between the MF6 arrays and the 1D sites (structured grid).

    shape:        (nlay, nrow, ncol) of the grid.
    top:          (nrow, ncol) elevation of the model top [m]; pressure heads are relative to it.
    node_user:    for each reduced MF6 node (index into X) its 0-based user node number.
    site_columns: site_id -> indices ``row * ncol + col`` of the model columns owned by the site.
    """
    shape: tuple[int, int, int]
    top: np.ndarray
    node_user: np.ndarray
    site_columns: dict[int, np.ndarray]

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
        """Heads X (reduced nodes) scattered onto the full (nlay, nrow, ncol) grid.

        Removed (IDOMAIN <= 0) and dry cells get NaN.
        """
        full = np.full(int(np.prod(self.shape)), np.nan)
        full[self.node_user] = x_reduced
        full[np.abs(full) >= DRY_HEAD_ABS_THRESHOLD] = np.nan
        return full.reshape(self.shape)

    def water_table(self, heads: np.ndarray) -> np.ndarray:
        """Head of the uppermost wet cell of every column, flattened to (nrow * ncol,).

        This is the water-table elevation in an unconfined model; NaN for columns without a
        wet cell.
        """
        wet = np.isfinite(heads)
        first_wet = np.argmax(wet, axis=0)
        table = np.take_along_axis(heads, first_wet[None], axis=0)[0]
        table[~wet.any(axis=0)] = np.nan
        return table.ravel()

    def site_pressure_heads(self, heads: np.ndarray) -> dict[int, float]:
        """Pressure head (water table - model top) [m] per 1D site, mean over its columns."""
        pressure = self.water_table(heads) - self.top.ravel()
        result = {}
        for site_id, cols in self.site_columns.items():
            values = pressure[cols]
            assert np.isfinite(values).any(), f"All model columns dry for 1D site_id={site_id}"
            result[site_id] = float(np.nanmean(values))
        return result

    def spread_recharge(self, contributions: dict[int, float]) -> np.ndarray:
        """Recharge [m/day] per model column, flattened to (nrow * ncol,).

        Every column gets the value of its owning 1D site; columns without a site get zero.
        """
        recharge = np.zeros(self.shape[1] * self.shape[2])
        for site_id, cols in self.site_columns.items():
            recharge[cols] = float(contributions[site_id])
        return recharge


def resolve_builder(spec: str) -> ModelBuilder:
    """Import the model builder given as "module:function" (the module must be importable)."""
    module_name, sep, attr = spec.partition(":")
    assert sep and module_name and attr, f"model_builder must be 'module:function', got {spec!r}"
    return getattr(importlib.import_module(module_name), attr)


def default_libmf6() -> Path:
    """libmf6 of the conda `modflow6` package in the environment the venv is based on."""
    name = "libmf6.dylib" if sys.platform == "darwin" else "libmf6.so"
    return Path(sys.base_prefix) / "lib" / name


class Model3DAPI:
    """3D backend running MODFLOW 6 in-process through the MODFLOW API (see the module doc).

    Config (``model_3d.common``):
      time_step_hours:  coupling step [h]; every coupling step is one MF6 time step
      model_builder:    "module:function" building the GWF model, see `ModelBuilder`
      model_builder_config: mapping passed to the builder (optional, empty by default)
      ims:              keyword arguments of flopy.mf6.ModflowIms (solver settings), e.g. {complexity: MODERATE}
      lib_path:         path to libmf6 (optional, `default_libmf6()` otherwise)

    Interface used by `Model3D.run_loop`, in this order: `build_cell_assignment()` (starts MF6),
    `initial_heads_to_1d()`, then per step `choose_dt()` and `model_step()`, `well_prediction()`,
    and `close()` at the end.
    """

    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        self.composed = composed
        self.cfg = model_3d_cfg
        self.locations_1d = [int(s) for s in locations_1d]
        hours = float(model_3d_cfg["time_step_hours"])
        self.time_step = np.timedelta64(int(round(hours * 3600)), "s")
        # Folder of the MF6 input and output files (MF6 working directory).
        self.sim_ws = Path(composed.workdir) / "model_3d"
        # Set by initialize(): the loaded library, grid bookkeeping, upper-case MF6 names and
        # the memory address of the heads.
        self.mf6: ModflowApi | None = None
        self.grid: GridMap | None = None
        self._model = self._recharge_pkg = None
        self._head_addr = None

    # ---- setup -----------------------------------------------------------
    def _time_discretization(self) -> tuple[float, int]:
        """TDIS of the whole simulated interval: (period length [days], number of time steps).

        One stress period split into equal time steps of the coupling step length, so every
        `prepare_time_step` advances MF6 by exactly one coupling step.
        """
        total = self.composed.end - self.composed.start
        n_steps, rest = divmod(total, self.time_step)
        assert n_steps > 0 and rest == np.timedelta64(0, "s"), (
            f"Simulated interval {total} must be a positive multiple of time_step {self.time_step}"
        )
        return float(total / np.timedelta64(1, "s")) / SECONDS_PER_DAY, int(n_steps)

    def _build_simulation(self) -> GwfSetup:
        """Create the MF6 input files with flopy.

        MFSimulation (file mfsim.nam) holds the time discretization TDIS, the model(s) and the
        solver. The builder adds the GWF model with its packages; the IMS solver package is
        then registered for that model. `write_simulation` writes all files to `sim_ws`;
        `exe_name` is unused since the executable is never started here.
        """
        self.sim_ws.mkdir(parents=True, exist_ok=True)
        sim = flopy.mf6.MFSimulation(sim_name="model_3d", sim_ws=str(self.sim_ws), exe_name="mf6")
        perlen, nstp = self._time_discretization()
        flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)])
        builder = resolve_builder(self.cfg["model_builder"])
        setup = builder(sim, self.locations_1d, dict(self.cfg.get("model_builder_config", {})))
        assert isinstance(setup, GwfSetup), f"model builder must return GwfSetup, got {type(setup)}"
        ims = flopy.mf6.ModflowIms(sim, **self.cfg["ims"])
        sim.register_ims_package(ims, [setup.gwf.name])
        sim.write_simulation(silent=True)
        return setup

    def initialize(self) -> None:
        """Write the input files, load libmf6 and initialize the simulation (called once).

        After `mf6.initialize()` all input is read and the heads X hold the starting heads.
        Then the memory address of X and the node mapping are looked up once.
        """
        assert self.mf6 is None, "Model3DAPI.initialize() called twice"
        setup = self._build_simulation()
        lib_path = Path(self.cfg["lib_path"]) if "lib_path" in self.cfg else default_libmf6()
        assert lib_path.exists(), f"libmf6 not found: {lib_path} (conda package modflow6, or set model_3d.common.lib_path)"
        LOG.info("[3D API] initializing MODFLOW 6 (%s) in %s", lib_path, self.sim_ws)
        self.mf6 = ModflowApi(str(lib_path), working_directory=str(self.sim_ws))
        self.mf6.initialize()

        # MF6 memory names are upper case.
        self._model = setup.gwf.name.upper()
        self._recharge_pkg = setup.recharge_package.upper()
        self._head_addr = self.mf6.get_var_address("X", self._model)
        node_user = np.asarray(self._get("NODEUSER", "DIS"), dtype=int)
        n_reduced = self.mf6.get_value_ptr(self._head_addr).size
        # MF6 allocates NODEUSER with a single entry when IDOMAIN removes no cells (reduced == user).
        node_user = np.arange(n_reduced) if node_user.size != n_reduced else node_user - 1
        self.grid = GridMap.from_setup(setup, node_user, self.locations_1d)

    def _get(self, var: str, component: str | None = None) -> np.ndarray:
        """Copy of the MF6 variable `var` of our model, optionally of a package (`component`)."""
        address = self.mf6.get_var_address(var, self._model, *([component] if component else []))
        return self.mf6.get_value(address)

    def _set_recharge(self, recharge_columns: np.ndarray) -> None:
        """Write the recharge [m/day] per model column into MF6's recharge package.

        The solver takes, for entry i of the package, the rate RECHARGE[i] on the cell NODELIST[i]
        (MF6 `rch_cf`: right-hand side = -recharge * cell area). NBOUND entries are in use;
        NODELIST holds reduced 1-based cell numbers, mapped here to model columns. MF6 may move
        an entry to the highest active cell of its column, so NODELIST is read at every step.
        RECHARGE is written through a pointer, directly into MF6 memory. BOUND must not be used:
        it has zero columns for this package (see the pitfall in the module doc).
        Must be called after `prepare_time_step`: MF6 reads the stress-period input there,
        which would overwrite earlier values. With a single stress period the input is read
        only once, so the values written here stay in force for the time step.
        """
        nbound = int(np.ravel(self._get("NBOUND", self._recharge_pkg))[0])
        nodes = np.asarray(self._get("NODELIST", self._recharge_pkg), dtype=int)[:nbound] - 1
        ncpl = self.grid.shape[1] * self.grid.shape[2]
        columns = self.grid.node_user[nodes] % ncpl
        address = self.mf6.get_var_address("RECHARGE", self._model, self._recharge_pkg)
        recharge = self.mf6.get_value_ptr(address)
        assert recharge.ndim == 1 and recharge.size >= nbound, f"Unexpected RECHARGE shape {recharge.shape}"
        recharge[:nbound] = recharge_columns[columns]

    def build_cell_assignment(self) -> None:
        """Start MF6; the name is the entry point `Model3D.run_loop` calls on every backend."""
        self.initialize()

    # ---- coupling --------------------------------------------------------
    def _pressure_heads(self) -> dict[int, float]:
        """Current heads X from MF6 memory, converted to pressure heads per 1D site."""
        heads = self.grid.full_heads(self.mf6.get_value_ptr(self._head_addr))
        return self.grid.site_pressure_heads(heads)

    def initial_heads_to_1d(self) -> dict[int, float]:
        """Pressure heads from the starting heads, sent to the 1D models before the first step."""
        assert self.mf6 is not None, "Model3DAPI.initialize() was not called"
        return self._pressure_heads()

    def choose_dt(self, current_time: np.datetime64, t_end: np.datetime64) -> np.timedelta64:
        """Next coupling step: always the fixed MF6 time step (TDIS cannot be changed later)."""
        remaining = t_end - current_time
        assert remaining >= self.time_step, f"Remaining {remaining} shorter than MF6 time step {self.time_step}"
        return self.time_step

    def model_step(self, dt: np.timedelta64, contributions) -> dict[int, float]:
        """Advance MF6 by one time step with the 1D recharge; return the new pressure heads.

        contributions: site_id -> recharge [m/day] computed by the 1D models for this step.
        """
        assert self.mf6 is not None, "Model3DAPI.initialize() was not called"
        dt_days = float(dt / np.timedelta64(1, "s")) / SECONDS_PER_DAY
        # 1. Start the time step. MF6 ignores the argument and takes the length from TDIS,
        #    so check that its step matches the coupling step.
        self.mf6.prepare_time_step(dt_days)
        mf6_dt = self.mf6.get_time_step()
        assert abs(mf6_dt - dt_days) < 1e-9 * max(1.0, dt_days), f"MF6 step {mf6_dt} d != coupling step {dt_days} d"
        # 2. Replace the recharge of this step by the 1D results.
        self._set_recharge(self.grid.spread_recharge(contributions))
        # 3. Solve the step; 4. write budgets and outputs (raises if the step did not converge).
        self.mf6.do_time_step()
        self.mf6.finalize_time_step()
        # 5. New heads -> pressure heads for the 1D models.
        return self._pressure_heads()

    def well_prediction(self, wells_dataset):
        """Water levels at monitoring wells; not implemented yet (milestone 7)."""
        _ = wells_dataset
        return {}

    def close(self) -> None:
        """Finalize MF6 (close its output files, free its memory) at the end of the simulation."""
        assert self.mf6 is not None, "Model3DAPI.initialize() was not called"
        self.mf6.finalize()
        self.mf6 = None
