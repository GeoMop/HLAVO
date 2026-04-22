# Status summary

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
