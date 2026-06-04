# Status summary

`2026-06-04`: `2a57707` @ `main` by `Codex <codex@openai.com>`

## Goal
Resolve the `STATUS.md` merge conflict by reconciling the local export-geometry handoff with the newer composed/logging handoff and keeping the status history consistent with the current worktree.

## Changes summary
- Resolved the conflict in [STATUS.md](/home/hlavo/workspace/STATUS.md) by keeping both branches' handoff entries, newest first, instead of dropping either the `2026-06-03` composed/logging work or the `2026-04-29` export-geometry work.
- Cleaned the `2026-06-03` status entry so it reflects the final local state rather than an intermediate reverted ParFlow attempt: [tests/soil_parflow/test_parflow_model.py](/home/hlavo/workspace/tests/soil_parflow/test_parflow_model.py) carries the interface fix, while [hlavo/soil_parflow/parflow_model.py](/home/hlavo/workspace/hlavo/soil_parflow/parflow_model.py) keeps the original production behavior.
- Removed merge-helper snapshots `STATUS_BACKUP_1616747.md`, `STATUS_BASE_1616747.md`, `STATUS_LOCAL_1616747.md`, and `STATUS_REMOTE_1616747.md` after folding their content into the main status history.

## Verified
- `grep -R -n '<<<<<<<\|=======\|>>>>>>>' STATUS*`
  result: conflict markers were present only in `STATUS.md` and `STATUS_BACKUP_1616747.md` before the merge cleanup.
- `git rev-parse --short HEAD`
  result: current `HEAD` is `2a57707` on `main`, used for this reconciliation checkpoint.

## Open items
- This reconciliation pass did not rerun runtime or pytest commands; it only merged the two diverged status histories and corrected the top handoff entry to match the present files.

`2026-06-03`: `4a5cf2e` @ `MS_distr` by `Codex`

## Goal
Fix `runs/composed_1d_only` visibility and noisy output problems while checking whether the apparent infinite loop is still non-advancing.

## Changes summary
- Tracked/uncommitted: [hlavo/misc/logging_utils.py](/home/hlavo/workspace/hlavo/misc/logging_utils.py) now owns the shared HLAVO logging setup, and [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py) plus [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py) call that common implementation instead of carrying duplicated handler/root-level logic.
- Tracked/uncommitted: [hlavo/kalman/model_1d.py](/home/hlavo/workspace/hlavo/kalman/model_1d.py) no longer branches on `KalmanMock`; `Model1D` always loads datasets through `Model1DData.from_config(...)`.
- Tracked/uncommitted: [tests/conftest.py](/home/hlavo/workspace/tests/conftest.py), [tests/composed/test_composed.py](/home/hlavo/workspace/tests/composed/test_composed.py), and [tests/composed/test_composed_config.yaml](/home/hlavo/workspace/tests/composed/test_composed_config.yaml) now create local zarr-backed mock profile/surface stores via a pytest fixture, inject the generated schema paths into a temporary runtime config, and keep the composed Dask test on a single mock 1D site.
- Tracked/uncommitted: [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py), [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py), and [hlavo/misc/logging_utils.py](/home/hlavo/workspace/hlavo/misc/logging_utils.py) now centralize HLAVO logging: root stdout stays `INFO`, the `hlavo` logger tree stays `DEBUG`, and per-site worker debug logs go to files such as `worker_1d_site_1.log` rather than the main `calculation.log`.
- Tracked/uncommitted: [tests/soil_parflow/test_parflow_model.py](/home/hlavo/workspace/tests/soil_parflow/test_parflow_model.py) was corrected to the current ParFlow wrapper contract instead of changing production behavior: `parflow.log` is checked for ParFlow success text, and `velocity` is expected as a face-centered `NZ + 1` array while `pressure` and `moisture` remain cell-centered `NZ` arrays. [hlavo/soil_parflow/parflow_model.py](/home/hlavo/workspace/hlavo/soil_parflow/parflow_model.py) keeps its original interface.
- Tracked/uncommitted: [runs/composed_1d_only/composed_config.yaml](/home/hlavo/workspace/runs/composed_1d_only/composed_config.yaml), [tests/model_1d/composed_config.yaml](/home/hlavo/workspace/tests/model_1d/composed_config.yaml), and [tests/model_1d/test_model_1d.py](/home/hlavo/workspace/tests/model_1d/test_model_1d.py) now use the config-defined two-hour simulation window `2025-03-06T00:00:00` to `2025-03-06T02:00:00`.
- Tracked/uncommitted: [tests/composed/test_composed_config.yaml](/home/hlavo/workspace/tests/composed/test_composed_config.yaml), [tests/composed/test_composed.py](/home/hlavo/workspace/tests/composed/test_composed.py), and [hlavo/kalman/model_1d.py](/home/hlavo/workspace/hlavo/kalman/model_1d.py) update the composed mock test to the current config schema and provide in-memory mock datasets for `KalmanMock`.
- Tracked/uncommitted: [hlavo/composed/model_3d.py](/home/hlavo/workspace/hlavo/composed/model_3d.py) lets the mock 3D backend use `model_3d.common.time_step_hours`, and [hlavo/kalman/kalman_state.py](/home/hlavo/workspace/hlavo/kalman/kalman_state.py) allows empty measurement structures to encode as an empty vector.
- Tracked/uncommitted: [hlavo/kalman/kalman.py](/home/hlavo/workspace/hlavo/kalman/kalman.py), [hlavo/kalman/model_1d.py](/home/hlavo/workspace/hlavo/kalman/model_1d.py), [hlavo/misc/auxiliary_functions.py](/home/hlavo/workspace/hlavo/misc/auxiliary_functions.py), and [hlavo/soil_parflow/parflow_model.py](/home/hlavo/workspace/hlavo/soil_parflow/parflow_model.py) replace raw print diagnostics with logging, summarize array/dataset DEBUG diagnostics, make ParFlow scratch directory names unique by process/thread/time, and report Kalman local target/completion progress.
- Tracked/uncommitted: [hlavo/composed/model_3d.py](/home/hlavo/workspace/hlavo/composed/model_3d.py) preserves mock backend `site_id -> head` mappings and returns the final loop time; [hlavo/composed/model_composed.py](/home/hlavo/workspace/hlavo/composed/model_composed.py) no longer converts that final time through `float()`.
- Staged before this coding work: [PLAN.md](/home/hlavo/workspace/PLAN.md) and [dev/.codex_docker/config.toml](/home/hlavo/workspace/dev/.codex_docker/config.toml) had pre-existing tracked edits. [PLAN.md](/home/hlavo/workspace/PLAN.md) now also contains this session's AGENT log and QaR notes.

## Verified
- `python3 -m py_compile hlavo/kalman/kalman_state.py hlavo/misc/logging_utils.py hlavo/main.py hlavo/composed/worker_1d.py` passed.
- `timeout 180s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only` passed after the shared logging refactor and the minimal `MeasurementsStructure.encode()` regression fix.
- `python3 -m py_compile hlavo/composed/model_composed.py hlavo/kalman/model_1d.py tests/conftest.py tests/composed/test_composed.py` passed.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 120s tests/run` passed after removing the `KalmanMock` branch and moving mock data setup into the fixture-backed local zarr store.
- `python3 -m py_compile hlavo/composed/model_3d.py hlavo/kalman/model_1d.py hlavo/kalman/kalman_state.py tests/composed/test_composed.py tests/model_1d/test_model_1d.py` passed.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 120s tests/run` passed: `1 passed`.
- `python3 -m py_compile hlavo/composed/worker_1d.py` passed after simplifying the worker-specific log routing.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run` passed after removing the worker `contextvars` filter: `1 passed`.
- `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/composed/worker_1d.py` passed after merging the worker file-handler helper into `logging_utils.py`.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run` passed after moving worker file-handler setup to the shared utility: `1 passed`.
- `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/composed/worker_1d.py hlavo/main.py` passed after switching the shared logging helpers to explicit logger parameters and `LevelFormatter()`.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run` passed after updating the logging call sites to `set_hlavo_loggers(...)`: `1 passed`.
- `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/main.py hlavo/composed/worker_1d.py` passed after collapsing the helper API back to fixed root/stdout and `hlavo`/debug destinations.
- `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run` passed after switching call sites to `ensure_debug_file_handler(...)`: `1 passed`.
- `python3 -m py_compile hlavo/soil_parflow/parflow_model.py tests/soil_parflow/test_parflow_model.py` passed after reverting the production-only ParFlow changes and moving the fix into the test.
- `PYTEST_ADDOPTS='tests/soil_parflow/test_parflow_model.py -q tests/model_1d/test_model_1d.py -q' timeout 240s tests/run` passed, confirming the face-centered `velocity` shape is still compatible with the model_1d / Kalman path.
- `PYTEST_ADDOPTS='tests/model_1d/test_model_1d.py -q' timeout 180s tests/run` passed: `1 passed, 3 skipped`.
- `timeout 180s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only` passed. Stdout showed exactly two 3D intervals (`00:00 -> 01:00`, `01:00 -> 02:00`), each with `[UKF] step 01/01 complete: parflow_evals=33, model_iterations=1320`, and final 3D time `2025-03-06T02:00:00`.
- `grep -E "botocore|s3fs|aiohttp" -n runs/composed_1d_only/calculation.log | tail -20` produced no output after narrowing DEBUG to `hlavo.*`.
- `grep -n "DEBUG.*hlavo.kalman" runs/composed_1d_only/calculation.log | tail -5` produced no output, while the same grep on `runs/composed_1d_only/worker_1d_site_1.log` showed Kalman DEBUG records.
- `tail runs/composed_1d_only/worker_1d_site_1.log` confirmed worker Kalman DEBUG records are written to the per-site worker log after separating worker file logging from `calculation.log`.
- `python3 -m py_compile hlavo/misc/auxiliary_functions.py hlavo/composed/worker_1d.py hlavo/kalman/model_1d.py hlavo/kalman/kalman.py hlavo/soil_parflow/parflow_model.py hlavo/composed/model_3d.py hlavo/composed/model_composed.py hlavo/main.py` passed.
- `timeout 70s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only` timed out while progressing, with stdout showing 1D worker start, `[UKF] Step 1/24`, step 1 completion, and step 2 start; no raw `value`, `data array`, or `sqrt func call` prints appeared.
- `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/main.py hlavo/composed/worker_1d.py hlavo/kalman/model_1d.py hlavo/kalman/kalman.py` passed after message-format changes.
- `timeout 65s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only` timed out while progressing, with stdout showing compact INFO lines such as `20:34:36 INFO  [UKF] step 01/24 target=2025-03-06T01:00:00` and `20:35:02 INFO  [UKF] step 01/24 complete: parflow_evals=33, model_iterations=1320`.
- `tail runs/composed_1d_only/calculation.log` confirmed worker Kalman DEBUG records are written and state arrays/datasets are summarized rather than dumped as full arrays.
- `timeout 300s tests/run` timed out after progress markers `F.....`; `timeout 180s env PYTEST_ADDOPTS='-x' tests/run` stopped at the first failure in `tests/composed/test_composed.py::test_setup_models_uses_real_dask_queues_with_config_mocks`.

## Open items
- The fixture-backed composed mock test passes, but it emits `zarr` warnings about `Uhelna` / `parflow` not being recognized as components of a Zarr hierarchy. The minimal local store layout is sufficient for the test but not warning-free.
- Integrated runs in this environment can warn that Dask port `8787` is already in use and pick another dashboard port. The run still succeeds.
- The completed `runs/composed_1d_only` process still reports `Unclosed client session` warnings from `aiohttp` at shutdown. The simulation succeeds, but cleanup of the underlying zarr/S3 or Dask client sessions should be reviewed separately.

`2026-04-29`: `cfad588` @ `main` by `Codex`

## Goal
Add reusable Paraview exports for built deep-model workspaces, including QGIS input surfaces and a terrain-conforming geometry grid, and verify them on `runs/composed_test/workdir`.

## Changes summary
- Unstaged: [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py) `export_geometry` now writes both the intermediate QGIS surfaces multiblock and the terrain-conforming geometry export when `model_3d.geometry` is present in the config.
- Unstaged: [hlavo/deep_model/build_modflow_grid.py](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py) now writes [surfaces.vtm](/home/hlavo/workspace/runs/composed_test/workdir/model_with_mine/surfaces.vtm) alongside `grid_materials.npz` during model build by reusing `qgis_reader.write_vtk_surfaces(...)`.
- Unstaged: [hlavo/deep_model/grid_to_paraview.py](/home/hlavo/workspace/hlavo/deep_model/grid_to_paraview.py) now computes curved node Z only from active columns and exports active cells only, so the final geometry is an active-only `UnstructuredGrid` `.vtu` without the distorted inactive shell at the outer boundary.
- Unstaged: [runs/composed_test/run_export_geometry.sh](/home/hlavo/workspace/runs/composed_test/run_export_geometry.sh) now uses `runs/run_0.sh` and targets `grid_materials.vtu`.
- Untracked: [tests/test_main_export_geometry.py](/home/hlavo/workspace/tests/test_main_export_geometry.py) covers the revised CLI export and verifies the curved grid contents through `pyvista`.
- Relevant generated artifacts: [surfaces.vtm](/home/hlavo/workspace/runs/composed_test/workdir/model_with_mine/surfaces.vtm) and [grid_materials.vtu](/home/hlavo/workspace/runs/composed_test/workdir/model_with_mine/grid_materials.vtu) were written from the existing testcase geometry.

## Verified
- `python3 -m py_compile hlavo/main.py hlavo/deep_model/build_modflow_grid.py hlavo/deep_model/grid_to_paraview.py tests/test_main_export_geometry.py`
  result: compile checks passed.
- `cd tests && PYTEST_ADDOPTS="test_main_export_geometry.py" bash ./run`
  result: `1 passed in 2.73s`.
- `python3 -m hlavo.main export_geometry runs/composed_test/config.yaml -w runs/composed_test/workdir -o runs/composed_test/workdir/model_with_mine/grid_materials.vtu`
  result: wrote [surfaces.vtm](/home/hlavo/workspace/runs/composed_test/workdir/model_with_mine/surfaces.vtm) and [grid_materials.vtu](/home/hlavo/workspace/runs/composed_test/workdir/model_with_mine/grid_materials.vtu).
- `python` + `pyvista.read(...)` on both outputs
  result: `surfaces.vtm` is readable as `MultiBlock` with 20 blocks; `grid_materials.vtu` is readable as active-only `UnstructuredGrid` with `1242182` cells and `cell_data=['materials', 'active', 'vtkOriginalCellIds']`.

## Open items
- Only the focused export test was run through `tests/run`; no broader project test pass is claimed.
- [runs/composed_test/run_export_geometry.sh](/home/hlavo/workspace/runs/composed_test/run_export_geometry.sh) now follows the repo rule and uses `runs/run_0.sh`, but local verification in this session had to call `python3 -m hlavo.main ...` directly because `run_0.sh` failed here with `docker not found on PATH`.

`2026-04-25`: `7e8010` @ `main` by `Jan Brezina <jan.brezina@tul.cz>`

## Goal
Finalize the `zf` dataset/store inspection CLI, expose it as an installed console command, and move ingress well-data tests and helpers onto explicit local `STORE_URL` overrides instead of schema rewriting or remote-store access.

## Changes summary
- `HEAD` is now `7e8010`; relevant committed work in this thread is present in [pyproject.toml](/home/hlavo/workspace/pyproject.toml), [hlavo/tools/zf.py](/home/hlavo/workspace/hlavo/tools/zf.py), [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py), [hlavo/tools/README.md](/home/hlavo/workspace/hlavo/tools/README.md), [hlavo/misc/aux_zarr_fuse.py](/home/hlavo/workspace/hlavo/misc/aux_zarr_fuse.py), [hlavo/ingress/well_data/well_data.py](/home/hlavo/workspace/hlavo/ingress/well_data/well_data.py), [hlavo/ingress/well_data/well_data_process.py](/home/hlavo/workspace/hlavo/ingress/well_data/well_data_process.py), and [tests/ingress/well_data/test_well_data.py](/home/hlavo/workspace/tests/ingress/well_data/test_well_data.py).
- The `zf` CLI is now available through packaging via `zf = "hlavo.tools.zf:main"` in `pyproject.toml`.
- [hlavo/tools/zf.py](/home/hlavo/workspace/hlavo/tools/zf.py) now uses `zarr_fuse.open_store(...)` only, matches a single full-path glob like `profiles.zarr/Uhelna/profiles*`, streams matched nodes via generator traversal, supports `-p` / `--print-dataset`, and reports per-store open errors without aborting the whole run.
- [hlavo/misc/aux_zarr_fuse.py](/home/hlavo/workspace/hlavo/misc/aux_zarr_fuse.py) now defaults shared dotenv loading to repo-root `.secrets_env`.
- [hlavo/ingress/well_data/well_data.py](/home/hlavo/workspace/hlavo/ingress/well_data/well_data.py) now accepts `STORE_URL` passthrough kwargs in `_remove_zarr_store()`, `_open_zarr_schema()`, `read_draw()`, and `read_sections_water_levels()`, so callers can operate on local zarr stores directly through `zarr_fuse` kwargs.
- [hlavo/ingress/well_data/well_data_process.py](/home/hlavo/workspace/hlavo/ingress/well_data/well_data_process.py) now imports plotting explicitly from `well_data_plot` instead of incorrectly calling `pdf_plot_all` through the `well_data` implementation module.
- [tests/ingress/well_data/test_well_data.py](/home/hlavo/workspace/tests/ingress/well_data/test_well_data.py) now uses temporary local `STORE_URL` paths, does not monkeypatch schema helpers, and normalizes `read_df()` results to pandas before pandas-specific assertions.
- Untracked but relevant: [tests/test_main_dataset.py](/home/hlavo/workspace/tests/test_main_dataset.py) is present in the working tree and covers the `zf` CLI against local test storages; it is not committed yet.

## Verified
- `python -m py_compile hlavo/tools/zf.py hlavo/main.py tests/test_main_dataset.py`
- `cd tests && PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH PYTEST_ADDOPTS="test_main_dataset.py" bash ./run`
  result: `3 passed`
- `python -m hlavo.tools.zf -p 'profiles.zarr/Uhelna/profiles'`
  confirmed `-p` prints the xarray dataset repr for a matched node.
- `python -m py_compile hlavo/ingress/well_data/well_data.py hlavo/ingress/well_data/well_data_process.py tests/ingress/well_data/test_well_data.py`
- `cd tests && PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH PYTEST_ADDOPTS="ingress/well_data/test_well_data.py" bash ./run`
  latest observed log state after the local `STORE_URL` and dataframe-type fixes: `tests/pytest.log` stayed at `..`; the previous `AttributeError: 'DataFrame' object has no attribute 'empty'` failure is fixed, but the final heavy water-level test did not complete within the waits used in this thread, so no full pass is claimed.
- `python -m py_compile hlavo/ingress/well_data/well_data_process.py`
  confirmed the `pdf_plot_all` import fix is syntactically valid.

## Open items
- `tests/ingress/well_data/test_well_data.py::test_borehole_water_level` remains slow/heavy in this environment; the local-store path and dataframe-type failure are fixed, but the thread did not wait for a full green completion.
- `zf` still emits the existing `hlavo_surface_schema.yaml` warning about `date_time` being defined in both `VARS` and `COORDS` when that schema is deserialized.
- [tests/test_main_dataset.py](/home/hlavo/workspace/tests/test_main_dataset.py) is still untracked.

`2026-04-24`: `d663eee` @ `main` by `Jan Brezina <jan.brezina@tul.cz>`

## Goal
Exercise the composed runtime through real Dask queues using config-selected mocks only, and leave `setup_models()` testable without monkeypatching internal runtime objects.

## Changes summary
- `HEAD` is now `d663eee` (`Test full mock model_composed.`).
- Committed: added [tests/composed/test_composed.py](/home/hlavo/workspace/tests/composed/test_composed.py) and [tests/composed/test_composed_config.yaml](/home/hlavo/workspace/tests/composed/test_composed_config.yaml).
- The composed test now calls [hlavo/composed/model_composed.py](/home/hlavo/workspace/hlavo/composed/model_composed.py) `setup_models()` directly, uses a real `LocalCluster` / `Client` and real Dask `Queue`s, and relies on config-selected mocks only (`KalmanMock` and `Model3DBackendMock`) rather than monkeypatching `Model3DConfig`, `Model3D`, or the 1D worker entrypoint.
- Relevant tracked files are currently clean; the worktree still contains many unrelated untracked files outside this task.

## Verified
- `cd tests && PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH PYTEST_ADDOPTS="composed/test_composed.py" bash ./run`
  result: `1 passed in 2.50s`
- `git log -1 --stat --summary --format=fuller`
  confirmed `d663eee` adds only the two composed test files.

## Open items
- The test covers real Dask queue orchestration with config-selected mocks, but it does not exercise the real 3D backend or a real one-step Kalman coupling API.
- The repository remains dirty because of unrelated untracked files; none of them were normalized here.

`2026-04-24`: `82c5c79` @ `main` by `Codex`

## Goal
Split the composed runtime into dedicated modules, remove direct 1D/3D model class selection from testcase configs, simplify `Model1D`, and verify the `runs/composed_3D_only` simulate path through `runs/run_0.sh`.

## Changes summary
- Staged, not committed: added [hlavo/composed/model_1d.py](/home/hlavo/workspace/hlavo/composed/model_1d.py), [hlavo/composed/model_3d.py](/home/hlavo/workspace/hlavo/composed/model_3d.py), and [hlavo/composed/model_composed.py](/home/hlavo/workspace/hlavo/composed/model_composed.py); [hlavo/composed/composed_model_mock.py](/home/hlavo/workspace/hlavo/composed/composed_model_mock.py) has been removed, and [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py), [hlavo/composed/__init__.py](/home/hlavo/workspace/hlavo/composed/__init__.py), and [hlavo/composed/run_composed.sh](/home/hlavo/workspace/hlavo/composed/run_composed.sh) now point to the split modules.
- Staged, not committed: [hlavo/composed/model_1d.py](/home/hlavo/workspace/hlavo/composed/model_1d.py) no longer carries `Model1DMock`; `Model1D` now only handles queue plumbing and Kalman selection, resolves `model_1d.kalman_class_name` via `hlavo.misc.class_resolve.resolve_named_class()`, and implements a local `KalmanMock` with a deterministic single-step fixed-velocity API.
- Staged, not committed: [hlavo/composed/model_3d.py](/home/hlavo/workspace/hlavo/composed/model_3d.py) now resolves the backend class from `backend_class_name`, and includes a minimal local `Model3DBackendMock`.
- Staged, not committed: [hlavo/deep_model/model_3d_cfg.py](/home/hlavo/workspace/hlavo/deep_model/model_3d_cfg.py) and [hlavo/deep_model/coupled_runtime.py](/home/hlavo/workspace/hlavo/deep_model/coupled_runtime.py) use `backend_class_name` with compatibility fallback from legacy `class_name`.
- Staged, not committed: [hlavo/kalman/__init__.py](/home/hlavo/workspace/hlavo/kalman/__init__.py) now exports `Kalman`, `KalmanFilter`, `KalmanFilterMock`, and `KalmanMock` names for resolver lookup.
- Staged, not committed: [runs/composed_3D_only/config.yaml](/home/hlavo/workspace/runs/composed_3D_only/config.yaml) no longer contains `model_1d.class_name` or `model_3d.common.class_name`; it now uses `model_1d.kalman_class_name: "KalmanMock"` and `model_1d.mock_velocity: 3.0e-4`.

## Verified
- `python -m py_compile hlavo/main.py hlavo/composed/__init__.py hlavo/composed/model_1d.py hlavo/composed/model_3d.py hlavo/composed/model_composed.py hlavo/kalman/__init__.py hlavo/deep_model/model_3d_cfg.py hlavo/deep_model/coupled_runtime.py`
  compile checks passed.
- `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`
  first failed at import time because `hlavo.composed.composed_model_mock` had been removed before entrypoints were rewired.
- `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`
  with the compatibility shim temporarily restored, the split runtime completed to `t=15.0`.
- `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`
  with `KalmanMock` and `mock_velocity: 0.0`, MODFLOW failed on the first step with `Simulation convergence failure` and exit code `1`.
- `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`
  with `KalmanMock` and `mock_velocity: 3.0e-4`, the testcase completed to `t=15.0`; final 1D worker results were `1D model 0 done`, `1D model 1 done`, and `1D model 2 done`.
- `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`
  rerun after removing both `model_1d.class_name` and `model_3d.common.class_name` from the testcase config also completed to `t=15.0`.

## Open items
- If `model_1d.kalman_class_name` is set to `Kalman` / `KalmanFilter`, `Model1D.step()` currently falls back to pressure-head passthrough because a real one-step Kalman coupling API is not wired yet.
- The 3D testcase still logs sanitization of many invalid heads in [hlavo/deep_model/coupled_runtime.py](/home/hlavo/workspace/hlavo/deep_model/coupled_runtime.py) (`759607` cells each step); this was observed but not debugged here.

`2026-04-21`: `347e41a` @ `Ot_modflow` by `Jan Brezina <jan.brezina@tul.cz>`

## Goal
Simplify config handling and composed/deep-model entrypoints, and align [hlavo/README.md](/home/hlavo/workspace/hlavo/README.md) with the current top-level CLI and workflow state.

## Changes summary
- `HEAD` is now `347e41a` (`Simplify config and compose`). Since the previous recorded status baseline `757aa51`, the branch also includes commit `085de9f` (`Combined config.yaml`).
- Committed work simplifies config handling and the composed runtime across [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py), [hlavo/composed/composed_model_mock.py](/home/hlavo/workspace/hlavo/composed/composed_model_mock.py), [hlavo/deep_model/build_modflow_grid.py](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py), [hlavo/deep_model/add_material_parameters.py](/home/hlavo/workspace/hlavo/deep_model/add_material_parameters.py), [hlavo/deep_model/run_model.py](/home/hlavo/workspace/hlavo/deep_model/run_model.py), [hlavo/deep_model/qgis_reader.py](/home/hlavo/workspace/hlavo/deep_model/qgis_reader.py), [hlavo/deep_model/model_3d_cfg.py](/home/hlavo/workspace/hlavo/deep_model/model_3d_cfg.py), [hlavo/misc/config.py](/home/hlavo/workspace/hlavo/misc/config.py), [hlavo/misc/class_resolve.py](/home/hlavo/workspace/hlavo/misc/class_resolve.py), and [runs/composed_3D_only/config.yaml](/home/hlavo/workspace/runs/composed_3D_only/config.yaml).
- [hlavo/README.md](/home/hlavo/workspace/hlavo/README.md) is committed in `347e41a` and now documents the current `build_model` / `simulate` top-level subcommands, their call paths, the deep-model artifact names, and the current composed coupling implementation.
- Relevant tracked files are currently clean, but the worktree still contains many unrelated untracked files and generated artifacts; no new staged or unstaged tracked changes were present when this status was updated.

## Verified
- `git log --oneline --decorate --no-merges 757aa51..HEAD`
  observed committed range: `085de9f Combined config.yaml`, `347e41a Simplify config and compose`
- `git show --stat --summary --format=fuller HEAD`
  confirmed `HEAD` metadata and the committed file set for `347e41a`
- `git log -1 --stat -- hlavo/README.md`
  confirmed the current [hlavo/README.md](/home/hlavo/workspace/hlavo/README.md) rewrite is part of `347e41a`
- `git status --short -- hlavo/README.md STATUS.md`
  confirmed no tracked modifications to those files at the time of this update

## Open items
- This status refresh did not rerun runtime or test commands; it only reconciled the handoff record against Git history and the current worktree.
- The repository remains dirty because of many untracked files outside this status task; they were not reviewed or normalized here.

`2026-04-12`: `757aa51` @ `Ot_modflow` by `Jan Brezina <jan.brezina@tul.cz>`

## Goal
Finish the composed 3D-only runtime/config cleanup, centralize class resolution, and simplify the CLI/build entrypoint path.

## Changes summary
- `HEAD` is now `757aa51` (`Simplify main.py`). Since the previous status baseline `8dd6469`, committed work simplified [hlavo/composed/composed_model_mock.py](/home/hlavo/workspace/hlavo/composed/composed_model_mock.py), [hlavo/composed/__init__.py](/home/hlavo/workspace/hlavo/composed/__init__.py), [hlavo/kalman/kalman.py](/home/hlavo/workspace/hlavo/kalman/kalman.py), [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py), [hlavo/deep_model/build_modflow_grid.py](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py), [runs/run_0.sh](/home/hlavo/workspace/runs/run_0.sh), [runs/composed_3D_only/config.yaml](/home/hlavo/workspace/runs/composed_3D_only/config.yaml), and [pyproject.toml](/home/hlavo/workspace/pyproject.toml).
- The composed runtime now uses config-driven class selection via a shared resolver, keeps `setup_models()` focused on orchestration, resolves 3D end time in `Model3D`, and uses fixed `model_with_mine` / `model_with_mine_work` testcase subdirs while keeping `model_3d.name` in config as the MODFLOW stem.
- `main.py` no longer writes an intermediate translated YAML for `build_model`; the build path now passes an in-memory translated config into [hlavo/deep_model/build_modflow_grid.py](/home/hlavo/workspace/hlavo/deep_model/build_modflow_grid.py).
- `pyproject.toml` now defines the `hlavo` console script, and `runs/run_0.sh` uses `python3 -m hlavo.main`.
- Uncommitted but relevant: [hlavo/misc/class_resolve.py](/home/hlavo/workspace/hlavo/misc/class_resolve.py) and [hlavo/misc/__init__.py](/home/hlavo/workspace/hlavo/misc/__init__.py) are still untracked in the working tree.

## Verified
- `python3 -m py_compile hlavo/main.py hlavo/deep_model/build_modflow_grid.py`
- `python3 -m py_compile hlavo/composed/composed_model_mock.py`
- `bash runs/run_0.sh --help`

## Open items
- The latest turn did not rerun the full `simulate` path after the final `pyproject.toml` / `run_0.sh` entrypoint refactor; only CLI/help and compile checks were re-verified after that step.
- `runs/composed_3D_only/config.yaml` still carries `measurements_config.measurements_file` for `Model1DMock`; thread note says this should likely become optional for the mock path.

`2026-04-12`: `8dd6469` @ `Ot_modflow` by `Jan Brezina <jan.brezina@tul.cz>`

## Goal
Make `runs/composed_3D_only` simulation runnable through `runs/run_0.sh` and document the `AGENT:` comment handling rule.

## Changes summary
- `HEAD` commit `8dd6469` (`Safer mf6 lookup adn inital pressure head treatment.`) updates [hlavo/composed/composed_model_mock.py](/home/hlavo/workspace/hlavo/composed/composed_model_mock.py) so startup can read initial heads from `uhelna.ic` when `uhelna.hds` does not exist yet.
- Staged, not committed: [runs/composed_3D_only/config.yaml](/home/hlavo/workspace/runs/composed_3D_only/config.yaml) now points to testcase-local model folders `model_with_mine` and `model_with_mine_work` while keeping the `AGENT:` comments intact.
- Staged, not committed: [AGENTS.md](/home/hlavo/workspace/AGENTS.md) now states that `AGENT:` comments are active developer instructions and must not be removed, rewritten, or moved unless implemented in the same change.
- Unstaged: [runs/run_0.sh](/home/hlavo/workspace/runs/run_0.sh) now uses `dev/venv-docker/bin/python`, resolves its real path, and prepends that interpreter directory to `PATH` so `mf6` is found without Python-side guessing.
- Unrelated dirty file still present: [dev/hlavo](/home/hlavo/workspace/dev/hlavo).

## Verified
- Ran `bash runs/run_0.sh simulate runs/composed_3D_only/config.yaml -w runs/composed_3D_only`.
- The run completed to `t=15.0` and produced `runs/composed_3D_only/model_with_mine_work/uhelna.hds`.
- Final 1D states reported by the run: `926.3612461459841`, `966.1570124097129`, `803.4881194643854`.

## Open items
- Measurement CSV resolution is still wrong for this testcase: the run looks for `runs/composed_3D_only/runs/column/...`, so Kalman setup is skipped.
- Working tree is not clean; the status above separates committed, staged, and unstaged changes only for files relevant to this thread.
