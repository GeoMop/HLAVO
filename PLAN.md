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
- Resolved `2026-06-03`: documented `ToyProblem` input and output file policy, config parameters, and single-run parametrization in `hlavo/soil_parflow/parflow_model.py`. Verified with `python3 -m py_compile hlavo/soil_parflow/parflow_model.py tests/soil_parflow/test_parflow_model.py`.


## AGENT log
- `2026-06-03`: Documented `ToyProblem` and `ToyProblem.run()` contracts in `hlavo/soil_parflow/parflow_model.py`, including config keys, fixed runtime files, CLM forcing requirements, run directory behavior, and Kalman-facing outputs.
- `2026-06-03`: Resolved `AGENTS.md` review answers from QaR: staging user edits is intentional but commits are forbidden unless explicitly requested; `AGENT` wording is the unified marker; `AGENT log` is for completed records while QaR is for unresolved user-facing inconsistencies; coding rules not present in `python_coding.md` were restored in `AGENTS.md`.
- `2026-06-03`: Reviewed `AGENTS.md` Workflow and in-source communication split. Removed duplicated staging, `AGENT` handling, large-change planning, and Python coding-rule references from `AGENTS.md`. No unresolved Workflow inconsistency remains after the cleanup.

## AGENT Questions And Remarks
