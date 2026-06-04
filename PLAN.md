## Curent goals


## Current Repository State

- `dashboard` - deployed, basic review functionality works fine
- `doc` - needs update
- `hlavo/ingress` - works, but missing processing of the detailed weather data
- `hlavo/composed` - need to finish refactoring
- `hlavo/deep_model` - finish refactoring to an implementation of the 3D model, not just standalone tool
- `hlavo/kalman` - need some reactoring to support both field simulations and laboratory simulations


## TODO points

### ToyProblem tests
- document ToyProblem in particular: input and output files policy, config parameters, how the single simulation run could be parametrised

### Fix infine loop in runs/composed_1d_only
It seems there is an infinite loop somewhere in the Kalman.
Introduce a single log for whole calculation and log start of individual parflow subprocesses and 
debugging array output. Use logging but with presence on the stdout for the Kalman time iterations.
We should know the time info with respect to the global time (from [3D]).

1. Do not report source file for INFO (only for DEBUG)
   Resolved `2026-06-03`: INFO records now use compact operator formatting without logger/source; DEBUG records keep logger and line number in `calculation.log`.
2. What number is after 'INFO'? Use fixed two digit space for consistent indenting. 
   Resolved `2026-06-03`: removed the process id from INFO formatting. The level column is fixed-width (`INFO `), and UKF step counters use two digits (`01/24`).
3. Shorten the line time, report the full time only as content of the first message, then just time to secons resolution only.
   Resolved `2026-06-03`: log prefix time is now `HH:MM:SS`; the first calculation message includes the full ISO timestamp in the message body.
4. Remove "Step 1/24 site_id=1 done update=applied" line
   Resolved `2026-06-03`: removed this message; update state is DEBUG-only.
5. Report global time only in Worker 1D or in 3D model, not down in the Kalman, report there only the next local time (not the previous time)
   Resolved `2026-06-03`: Kalman INFO reports only the local target time, e.g. `[UKF] step 01/24 target=2025-03-06T01:00:00`; global interval remains in 1D/3D messages.
6. Report next/target step local time at the beginning of the Kalman sime step, after completition report number of Parflow evaluation and its total number of iterations 
   (that was available in DEBUG)
   Resolved `2026-06-03`: Kalman completion now reports `parflow_evals=33, model_iterations=1320` for the observed first one-hour UKF step.
7. Why just single day simulation in run/composed_1d_only runs so long while tests/model_1d/test_model_1d.py (which should be effectively same configuration) runs fast even for 3 day simulation?
   Answer `2026-06-03`: `tests/model_1d/test_model_1d.py` does not run a three-day Kalman step. Its config has `end_datetime: 2025-03-09`, but the active `test_step()` calls `model.step(start=2025-03-06T00:00:00, target=2025-03-06T02:00:00)`, so it exercises only two local UKF hours and bypasses the 3D/Dask orchestration. The integrated `runs/composed_1d_only` run sends one 3D day to 1D, which creates 24 local UKF hourly steps. Each one-hour UKF step performs 33 ParFlow state-transition evaluations with 40 internal ParFlow model iterations each, i.e. 1320 ParFlow model iterations per UKF hour and about 31,680 for the one-day window. The current smoke showed step 1/24 taking about 26 s in this environment, so the full one-day run is expected to be many minutes rather than comparable to the targeted two-hour unit test.
   AGENT: So modify the unit test to actualy use the time form config file and unify both config files to simulate 2 hours only. Make the runs test config to run Kalman for 1 h simulation only performing 2 3D model loop iterations.
   Also there is an error:
   ```
   21:00:01 INFO  [UKF] step 24/24 complete: parflow_evals=33, model_iterations=1320
2026-06-03 21:00:01,523 - distributed.worker - ERROR - Compute Failed
Key:       model1d_worker_entry-0fb80f2105cda5880940cc2d53c4b650
State:     long-running
Task:  <Task 'model1d_worker_entry-0fb80f2105cda5880940cc2d53c4b650' model1d_worker_entry(...)>
Exception: "AssertionError('')"
Traceback: '  File "/home/hlavo/workspace/hlavo/composed/worker_1d.py", line 127, in model1d_worker_entry\n    return model.run_loop(queue_name_in, queue_name_out)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/hlavo/workspace/hlavo/composed/worker_1d.py", line 89, in run_loop\n    velocity = self.model.step(\n               ^^^^^^^^^^^^^^^^\n  File "/home/hlavo/workspace/hlavo/kalman/model_1d.py", line 173, in step\n    darcy_velocity = self.kalman.kalman_step(\n                     ^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/hlavo/workspace/hlavo/kalman/kalman.py", line 1054, in kalman_step\n    self.test_measurements_struc.encode(measurements_test_dict)\n  File "/home/hlavo/workspace/hlavo/kalman/kalman_state.py", line 517, in encode\n    assert value_dict, ""\n           ^^^^^^^^^^\n'
 
   ```
   Resolved `2026-06-03`: `runs/composed_1d_only` and `tests/model_1d` configs now cover only `2025-03-06T00:00:00` to `2025-03-06T02:00:00`; the model_1d test reads that window from config; the mock 3D backend can use `time_step_hours: 1`, producing two one-hour 3D loops; the composed mock test config was updated to the current schema and in-memory mock 1D data path; empty `test_measurements` structures now encode as an empty vector, fixing the reported Kalman assertion.

## AGENT log
- `2026-06-04`: Refined the shared logging API in [hlavo/misc/logging_utils.py](/home/hlavo/workspace/hlavo/misc/logging_utils.py) and updated its call sites in [hlavo/main.py](/home/hlavo/workspace/hlavo/main.py) and [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py). Removed the `calculation_formatter()` wrapper in favor of direct `LevelFormatter()` use, renamed `configure_hlavo_root_logging()` to `set_hlavo_loggers(...)`, and made both handler helpers take the destination logger explicitly. Verified with `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/composed/worker_1d.py hlavo/main.py` and `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run`.
- `2026-06-04`: Merged the worker-only file-handler helper into [hlavo/misc/logging_utils.py](/home/hlavo/workspace/hlavo/misc/logging_utils.py). `ensure_file_handler(...)` now documents its root-vs-named-logger behavior and supports `logger_name="hlavo"` for HLAVO-only files; [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py) now uses that shared helper instead of a local copy. Verified with `python3 -m py_compile hlavo/misc/logging_utils.py hlavo/composed/worker_1d.py` and `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run`.
- `2026-06-04`: Simplified [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py) worker log routing. Per-site worker files are still kept separate, but the worker-specific `contextvars` filter was removed; each 1D worker now adds its file handler directly to the `hlavo` logger tree. Verified with `python3 -m py_compile hlavo/composed/worker_1d.py` and `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 180s tests/run`.
- `2026-06-03`: Moved the common main/worker logging setup into [hlavo/misc/logging_utils.py](/home/hlavo/workspace/hlavo/misc/logging_utils.py). `main.py` and `worker_1d.py` now share one implementation for root log levels, stdout INFO handler policy, and DEBUG file-handler setup; only the worker site filter remains worker-specific. Verified with `python3 -m py_compile ...`, `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 120s tests/run`, and `timeout 180s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only`.
- `2026-06-03`: Removed the `KalmanMock`-specific branch from [hlavo/kalman/model_1d.py](/home/hlavo/workspace/hlavo/kalman/model_1d.py). The composed mock test now creates local zarr-backed profile/surface stores through a pytest fixture in [tests/conftest.py](/home/hlavo/workspace/tests/conftest.py), injects schema paths into a temporary runtime config, and runs through the same `Model1DData.from_config(...)` path as production. Verified with `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 120s tests/run`.
- `2026-06-03`: Changed 1D worker file logging from the main `calculation.log` to per-site worker files (`worker_1d_site_<site_id>.log`) with a worker-context filter, so parallel workers do not interleave DEBUG logs in the main log or each other's files.
- `2026-06-03`: Reviewed the debugging-related changes and narrowed retained logging to HLAVO loggers only. Added separate-line comments justifying the remaining non-obvious changes for bounded test windows, mock 3D timestep control, mock 1D datasets, and empty measurement encoding.
- `2026-06-03`: Completed the follow-up `runs/composed_1d_only` fix. The integrated run now finishes in two one-hour 3D loops with one UKF local hour per loop, and the previous empty `test_measurements` assertion no longer occurs. Targeted verification: `python3 -m py_compile ...`, `PYTEST_ADDOPTS='tests/composed/test_composed.py -q' timeout 120s tests/run`, `PYTEST_ADDOPTS='tests/model_1d/test_model_1d.py -q' timeout 180s tests/run`, and `timeout 180s bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only`.
- `2026-06-03`: Processed `runs/composed_1d_only` PLAN items 1-7. Added compact INFO/detailed DEBUG formatter, removed INFO process id/source details, shortened log prefix timestamps, simplified Kalman INFO to local target-only messages, reported ParFlow evaluation/model-iteration counts at UKF step completion, and answered why the integrated one-day run is much slower than `tests/model_1d/test_model_1d.py`.
- `2026-06-03`: Resolved `AGENTS.md` review answers from QaR: staging user edits is intentional but commits are forbidden unless explicitly requested; `AGENT` wording is the unified marker; `AGENT log` is for completed records while QaR is for unresolved user-facing inconsistencies; coding rules not present in `python_coding.md` were restored in `AGENTS.md`.
- `2026-06-03`: Reviewed `AGENTS.md` Workflow and in-source communication split. Removed duplicated staging, `AGENT` handling, large-change planning, and Python coding-rule references from `AGENTS.md`. No unresolved Workflow inconsistency remains after the cleanup.

## AGENT Questions And Remarks
- `2026-06-04`: The requested logger renaming cannot be completed as a local cleanup only. Python's actual root logger keeps its built-in identity, and the repository still emits through the `hlavo.*` logger tree. This turn made handler ownership explicit via `HLAVO_ROOT_LOGGER` and `HLAVO_DEBUG_LOGGER` objects, but a true move to a separate `hlavo_debug.*` tree would require a broader repo-wide logger-name migration.
- `2026-06-04`: The requested pre-write staging step could not be completed for this turn because `.git/index.lock` already existed in the repository and `git add ...` failed before the edits started. The code changes and verification completed, but the worktree should be checked for the stale/active git lock before the next staging step.
- `2026-06-04`: The simplified worker logging in [hlavo/composed/worker_1d.py](/home/hlavo/workspace/hlavo/composed/worker_1d.py) now relies on the current Dask deployment shape: one worker process per 1D worker when strict per-site file separation is required. The composed Dask test still passes, but a future same-process threaded execution mode would need different routing again.
- `2026-06-03`: The fixture-backed composed mock test passes, but `zarr` emits warnings such as `Object at Uhelna is not recognized as a component of a Zarr hierarchy.` when opening the minimal local stores through `zarr_fuse`. The datasets load and the test passes, but the fixture store layout is only the minimal working shape, not a warning-free hierarchy.
- `2026-06-03`: Integrated runs can emit a Dask warning that port `8787` is already in use, after which Dask selects another dashboard port. The simulation still succeeds; this is only a diagnostic warning in this environment.
- `2026-06-03`: The completed `runs/composed_1d_only` process still reports `Unclosed client session` warnings from `aiohttp` at shutdown. The simulation succeeds, but the underlying zarr/S3 or Dask client-session cleanup should be reviewed separately.
