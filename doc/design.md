# HLAVO Tentative Design

This note is a tentative target design for the main runtime structure. It is
not only a description of the current code. Some naming and responsibility
splits below are recommendations for later refactoring once the active branches
are merged and the interfaces are stable again.

The project is organized mostly around runtime boundaries and data passing, not
around deep inheritance hierarchies. The essential pieces are the run context,
the 1D and 3D runtime wrappers, the Kalman engine, the forward models, and the
message/data carriers between them.

## Top-level runtime structure

### [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py)
- `main()`: CLI entrypoint for the user-visible commands.
- Main subcommand dispatch:
  - `build_model` -> [_run_build_model()](/home/hlavo/workspace/hlavo/main.py:64) -> [build_model()](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py:69)
  - `simulate` -> [_run_simulate()](/home/hlavo/workspace/hlavo/main.py:70) -> [run_simulation()](/home/hlavo/workspace/hlavo/composed/model_composed.py:82)
  - `dataset` -> [_run_dataset()](/home/hlavo/workspace/hlavo/main.py:77)
- Desired responsibility: only CLI parsing, workdir resolution, and top-level logging policy.

### [hlavo/composed/model_composed.py](/home/hlavo/workspace/hlavo/composed/model_composed.py)
- `run_simulation()`: top-level coupled run driver. It owns the Dask cluster lifecycle.
- `setup_models()`: creates [ComposedData](/home/hlavo/workspace/hlavo/composed/common_data.py:6), instantiates the 3D side through [Model3D](/home/hlavo/workspace/hlavo/composed/model_3d.py:59), and submits 1D workers through [model1d_worker_entry()](/home/hlavo/workspace/hlavo/composed/worker_1d.py:88).
- `_model_1d_site_ids()`: extracts the worker site set from config.
- Desired responsibility: orchestration only. It must know the final site list because worker startup requires that information before any site-local model is constructed.
- Design quality: acceptable. The file is the right place for process and queue orchestration.
- Weak point: the file should only consume one normalized `model_1d` site-list shape; keeping compatibility logic for multiple shapes here will spread config migration work into orchestration code.
  AGENT: The site_ids are minimal information that has to be available at the top level in order to start the workers.
  So be more specific about this weak point, I do not understand your formulation.
  Resolved: Narrowed the weak point to config-shape compatibility logic, not to the need to know the site list itself.

  

### Target class split for the coupled runtime
- Current naming mixes orchestration and physical-model ownership on the 3D side.
- Recommended target naming:
  - current [hlavo/composed/model_3d.py::Model3D](/home/hlavo/workspace/hlavo/composed/model_3d.py:59) -> `Master3D` 
  - current [hlavo/deep_model/coupled_runtime.py::Model3DBackend](/home/hlavo/workspace/hlavo/deep_model/coupled_runtime.py) -> `Model3D`
- Desired split:
  - composed runner: queue protocol, timestep synchronization, multi-site aggregation
  - 3D model: groundwater state, cell assignment, timestep choice, model step
  - config layer: normalize config once and provide one stable input shape to both
- Weak point: as long as orchestration and 3D-model naming stay inverted, future refactors will remain harder to reason about and review.

## Shared run context and coupling messages

### [hlavo/composed/common_data.py](/home/hlavo/workspace/hlavo/composed/common_data.py)
- `ComposedData`: shared runtime context for the coupled simulation.
- Responsibility: seed, start/end datetimes, workdir, config directory.
- Design quality: good. Small, explicit, and easy to reason about.

### [hlavo/composed/data_3d_to_1d.py](/home/hlavo/workspace/hlavo/composed/data_3d_to_1d.py)
- `Data3DTo1D`: message from 3D to 1D.

### [hlavo/composed/data_1d_to_3d.py](/home/hlavo/workspace/hlavo/composed/data_1d_to_3d.py)
- `Data1DTo3D`: message from 1D to 3D.
- Responsibility: target time, site id, coordinates, bottom Darcy velocity.
- Design quality: good. The coupling contract is visible in one place.
- Weak point: internal message semantics are checked only by runtime assertions at the queue boundary, not by a dedicated validation layer.
  The messages are only internal so no need for extensive versioning and compatibility handling.

## 1D model and assimilation side

### [hlavo/kalman/model_1d.py](/home/hlavo/workspace/hlavo/kalman/model_1d.py)
- `Model1D`: main 1D runtime wrapper used by the composed run.
- `Model1DData`: owns site-specific profile and surface datasets.
- `Model1DLocation`: site location carrier.
- `KalmanMock`: mock implementation of the Kalman-side behavior for tests.
- `create_kalman_measurements_config()`: converts dataset metadata into Kalman measurement config.
- Construction graph:
  - [Worker1D](/home/hlavo/workspace/hlavo/composed/worker_1d.py:29) -> [Model1D.from_config()](/home/hlavo/workspace/hlavo/kalman/model_1d.py:150)
  - `Model1D` -> [KalmanFilter.from_config()](/home/hlavo/workspace/hlavo/kalman/kalman.py:58) or `KalmanMock.from_config()`
- Desired responsibility: one site-local runtime wrapper that binds together loaded data, one Kalman-like engine, and one `step()` interface for the coupled runner.
- Design quality: mixed. `Model1D` is the right abstraction boundary, but the file still combines data loading, mock dataset generation, config rewriting, and runtime stepping.
- Weak point: configuration mutation inside `Model1D.from_config()` and `create_kalman_measurements_config()` makes the object harder to test and makes future config normalization brittle.

### [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py)
- `Worker1D`: Dask worker-side wrapper around `Model1D`.
- `model1d_worker_entry()`: worker entrypoint submitted by [setup_models()](/home/hlavo/workspace/hlavo/composed/model_composed.py:26).
- Responsibility: receive 3D messages, run local 1D step, send recharge/velocity back.
- Desired responsibility restriction: no model-specific config normalization beyond what is required to build one `Model1D`.
- Design quality: good. Runtime responsibilities are focused and operationally clear.

### [hlavo/kalman/kalman.py](/home/hlavo/workspace/hlavo/kalman/kalman.py)
- `KalmanFilter`: main assimilation engine.
- `from_config()`: constructor from config source.
- `run()`: full UKF execution loop.
- Responsibility: configure the forward model, own UKF state and measurements, run assimilation, save results.
- Desired responsibility restriction: remain the assimilation engine, not the place where unrelated file resolution or coupled-run orchestration accumulates.
- Design quality: functionally central but too large.
- Weak point: the class size and mixed concerns will make future numerical changes, debugging, and regression isolation increasingly expensive.

### [hlavo/kalman/kalman_state.py](/home/hlavo/workspace/hlavo/kalman/kalman_state.py)
- `StateStructure`: schema of the model state vector.
- `MeasurementsStructure`: schema of the observation vector.
- `Measure`, `GVar`, `GField`, `CalibrationCoeffs`: state and measurement parameterization helpers.
- Desired responsibility: isolate state-layout and observation-layout definitions from runtime execution.
- Design quality: good. This is one of the cleaner separations in the repo.

## 3D model and deep-model side

### [hlavo/composed/model_3d.py](/home/hlavo/workspace/hlavo/composed/model_3d.py)
- Current class: `Model3D`
- Target role: this should become the coupled 3D worker/runner, not the physical 3D model abstraction.
- Main methods:
  - constructor chooses the backend class
  - `run_loop()` drives the 3D/1D synchronization loop
- Construction graph:
  - [setup_models()](/home/hlavo/workspace/hlavo/composed/model_composed.py:26) -> [Model3D()](/home/hlavo/workspace/hlavo/composed/model_3d.py:59)
  - `Model3D` -> backend chosen by [resolve_named_class()](/home/hlavo/workspace/hlavo/composed/model_3d.py:64)
- Desired responsibility restriction: queue protocol and timestep synchronization only; physical state evolution should stay in the backend model class.
- Design quality: acceptable. The wrapper is structurally small and focused.
- Weak point: backend configuration ownership is still split between this wrapper and the deep-model config layer, which already caused config-shape regressions.

### [hlavo/deep_model/model_3d_cfg.py](/home/hlavo/workspace/hlavo/deep_model/model_3d_cfg.py)
- `Model3DCommonConfig`: normalized common 3D config object.
- `resolve_model_3d_section()`, `resolve_model_3d_common_raw()`: config resolution layer.
- Desired responsibility: one authoritative normalization layer for the 3D runtime/build config shape.
- Design quality: good in intent.
- Weak point: this layer still overlaps conceptually with the composed 3D wrapper, so config authority is not yet fully unified.

### [hlavo/deep_model/build_modflow_grid.py](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py)
- `BuildConfig`: build-time configuration object for the deep model.
- `build_model()`: main 3D model build entrypoint.
- Responsibility: turn GIS, geometry, and material inputs into a runnable MODFLOW workspace.
- Design quality: good. Build-time concerns are reasonably separated from runtime coupling.
- Weak point: build-time config and runtime config still have adjacent but not fully unified ownership, which can produce subtle divergence in future model evolution.

## Forward model implementations

### [hlavo/soil_parflow/parflow_model.py](/home/hlavo/workspace/hlavo/soil_parflow/parflow_model.py)
- `ToyProblem`: ParFlow-based 1D Richards forward model.
- Responsibility: configure a ParFlow run, execute it, and expose outputs for assimilation.
- Design quality: acceptable. It wraps a difficult external dependency behind one main class.
- Weak point: file I/O policy, ParFlow process control, and physical configuration live in one place, so maintenance risk is high whenever the ParFlow interface changes.

### [hlavo/soil_py/richards.py](/home/hlavo/workspace/hlavo/soil_py/richards.py)
- `RichardsEquationSolver`: pure-Python Richards solver.
- `RichardsSolverOutput`: structured output object.
- Design quality: good as an isolated numerical component.
- Weak point: if this solver is expected to stay aligned with the ParFlow model, the repo currently has no obvious single contract layer enforcing parity between them.

### [hlavo/soil_py/soil.py](/home/hlavo/workspace/hlavo/soil_py/soil.py)
- `VanGenuchtenParams`
- `SoilMaterialManager`
- Responsibility: soil hydraulic parameter model and material property evaluation.
- Design quality: good. Domain physics are kept in a dedicated module instead of being scattered through solvers.

## Runtime data ingress

### [hlavo/ingress/moist_profile/load_zarr_data.py](/home/hlavo/workspace/hlavo/ingress/moist_profile/load_zarr_data.py)
- `load_measurments_data()`
- `load_meteo_data()`
- Responsibility: runtime boundary for loading profile and meteorological datasets from Zarr-backed storage.
- Desired responsibility restriction: only dataset loading and root-node addressing, not broader schema interpretation.
- Design quality: good for runtime simplicity.
- Weak point: hard-coded storage paths inside loader functions will become a maintenance problem if dataset layout evolves or multiple deployments need different schema roots.

## Absolute core set

If only a few pieces are kept in mind, these are the structural core of the project:

1. `ComposedData`
2. `Model1D`
3. `KalmanFilter`
4. current `Model3D` runner, later preferably renamed to `Worker3D` or `Coupled3DRunner`
5. current `Model3DBackend`, later preferably the actual `Model3D`
6. `Model3DCommonConfig` / `BuildConfig`
7. `ToyProblem`
8. `Data3DTo1D` and `Data1DTo3D`

Together they define:
- global run context
- 1D site-local state and data ownership
- assimilation state and execution
- 3D runtime stepping and synchronization
- 3D config/build ownership
- forward-model execution
- 1D/3D coupling contract

## Refactoring timing

The naming and responsibility shifts above should be treated as deferred design
targets. Deeper refactoring should wait until the currently diverged branches
are merged and the config interfaces stop moving.
