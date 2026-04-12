# Status summary

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
