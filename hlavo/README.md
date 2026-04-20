# HLAVO package overview

Top-level package for the hydrology modeling stack. The active top-level CLI is
[`hlavo/main.py`](/home/hlavo/workspace/hlavo/main.py).

## Top-level scripts

- `main.py` - current package CLI with the active `build_model` and `simulate`
  subcommands
- `pbs_run.sh` - helper script for test composed simulation runs

## Subdirectories

- `composed` - coupled runtime wiring between 1D sites and the 3D deep model;
  currently contains the Dask-based mock integration used by `simulate`
- `deep_model` - MODFLOW 6 based deep vadose zone model build, run, and
  visualization tools together with model resources
- `ingress` - data loading and preprocessing utilities for meteorological,
  moisture-profile, and well data
- `kalman` - Unscented Kalman Filter based 1D assimilation driver and result
  handling
- `misc` - shared config loading and utility helpers
- `soil_parflow` - ParFlow based 1D/surface model classes used by the Kalman
  workflow
- `soil_py` - pure Python Richards solver experiments; not part of the main
  integrated workflow

## Entry script `main.py`

Current top-level subcommands and their call paths:

- `build_model <config_file> [-w <workdir>]`
  - CLI entry: [`hlavo/main.py`](/home/hlavo/workspace/hlavo/main.py)
    `_run_build_model()`
  - main implementation:
    [`hlavo/deep_model/build_modflow_grid.py`](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py)
    `build_model(config_source, workspace)`
  - internal build chain:
    `build_model()`
    -> `BuildConfig.from_source()`
    -> `build_modflow_grid()`
    -> `write_material_model_files()`
    -> `write_modflow_inputs()`
  - purpose: read GIS/config inputs, derive grid geometry and materials, then
    write MODFLOW input files into the resolved model workspace

- `simulate <config_file> [-w <workdir>]`
  - CLI entry: [`hlavo/main.py`](/home/hlavo/workspace/hlavo/main.py)
    `_run_simulate()`
  - main implementation:
    [`hlavo/composed/composed_model_mock.py`](/home/hlavo/workspace/hlavo/composed/composed_model_mock.py)
    `run_simulation(work_dir, config_path)`
  - internal runtime chain:
    `run_simulation()`
    -> `setup_models()`
    -> `Model3DConfig.from_source()`
    -> `_parse_locations()`
    -> Dask worker submission through `model1d_worker_entry()`
    -> `Model3D.run_loop()`
    -> `Model3DBackend.model_step()`
  - current behavior: starts a local Dask cluster, runs one mock/selected 1D
    worker per configured site, exchanges `Data3DTo1D` / `Data1DTo3D` messages,
    and advances the copied MODFLOW workspace step-by-step

## Deep model notes

The top-level CLI currently exposes only the combined `build_model` workflow.
Within [`hlavo/deep_model`](/home/hlavo/workspace/hlavo/deep_model) the
standalone scripts still exist and are useful for direct iteration:

- `build_modflow_grid.py` - geometry export plus MODFLOW input writing
- `add_material_parameters.py` - material and unsaturated parameter arrays
- `run_model.py` - run an already prepared MODFLOW model and export plan-view /
  ParaView outputs
- `visualize_results.py` - postprocess an existing MODFLOW run into plots and
  ParaView datasets

Current fixed workspace artifacts used by the deep-model code:

- model workspace directory default:
  `workdir/model_with_mine` unless overridden by config/workdir resolution
- build outputs:
  `grid_materials.npz`, `material_parameters.npz`
- ParaView outputs:
  `uhelna_results.vtu`, `uhelna_materials.vtr`
- plot directory:
  `plots/`
- plan-view plot outputs from `run_model.py`:
  `grid_active_mask.pdf`, `idomain_top.pdf`, `head_groundplan.pdf`,
  `groundwater_surface.pdf`, `velocity_groundplan.pdf`,
  `materials_x_section.pdf`, `materials_y_section.pdf`
- additional temporal/material-class plots from `visualize_results.py`:
  `material_class_x_section.png`, `material_class_y_section.png`, plus
  config-driven groundwater change / hydrograph / time-section plot names

## Coupling notes

- The current top-level `simulate` path is implemented in
  `composed/composed_model_mock.py`; it is still a development integration layer
  and keeps a mock 1D model implementation by default.
- If `model_1d.class_name` is configured, the worker side resolves the class by
  name via `hlavo.misc.class_resolve.resolve_named_class()`.
- The 3D side uses
  [`hlavo/deep_model/coupled_runtime.py`](/home/hlavo/workspace/hlavo/deep_model/coupled_runtime.py)
  to copy the prepared MODFLOW workspace, rewrite recharge and TDIS files per
  step, run `mf6`, sanitize heads, and feed averaged heads back to 1D workers.
