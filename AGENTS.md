# CODEX context ofr HLAVO project

## Project summary

HLAVO is a hydrology modeling system focused on predicting groundwater table
dynamics from meteorological inputs and soil moisture profile measurements. The
repo combines data ingress pipelines, surface modeling (Richards/ParFlow) with
Kalman-filter assimilation, and deep vadose zone model generation from GIS into
MODFLOW 6 inputs, with runs/configs under `runs/`.

## Project summary (short)

Hydrology modeling stack for the Uhelná locality: ingest meteo + soil moisture,
assimilate in surface models, and build deep vadose zone MODFLOW 6 models from
GIS sources.

## CODEX Ignore Folders
- `DeepSensor`
- `microchip_SW`
- `notebooks`
- `_*`

## CODEX Readonly Folders
- `dev`
- `hlavo/deep_model/GIS`

## Project Structure
- `hlavo` - sources of the digital twin prediction system
- `tests` - pytest based unit tests of individual source blocks
- `runs`  - integrated tests and production configurations

## `hlavo/` Project Source Structure

- `composed/` - communication of deep_model and kalman (surface model) in a parallel run
- `deep_model/` - model of the deep vadoze zone
- `ingress/` - data ingress tools and services
- `kalman/` - surface model assimilation (using profile measurements)
- `soil_parflow/` - ParFlow based surface model (Richards)
- `soil_py/` - Pure Python minimalistic Richards' solver.

## `runs` Guidelines
- User will use bash scripts `runs/run.sh` ans `runs/run_c.sh`. Directly or through custom scripts in the 
  individual testcase subdirs.
- CODEX already run in the docker environment so you should use `runs/run_0.sh` exclusively and mimic the 
  user custom scripts. E.g. if you are asked to test through `build_model.sh` in the `runs` subdir, you should 
  inspect the custom script and mimic its call using `run_0.sh`.


## CODEX Guidelines
- treat keyword 'AGENT:' in comments as a source context dependent message for your further development
- any comment containing `AGENT:` is an active developer instruction
- NEVER remove, rewrite, or move an `AGENT:` comment unless you implement that instruction in the same change
- if you make only a local fix around an `AGENT:` comment, leave the comment untouched
- if an `AGENT:` instruction looks outdated or wrong, ask before removing it
- Always review your changes before finishing for human review.

## Status tracking
- `STATUS.md` is the handoff log for interrupted or multi-turn work. Update it when a session ends with unfinished relevant work, when the user asks for a status review, or when a fresh checkpoint would help the next session continue without re-discovery.
- Keep newest entry first. Do not rewrite older entries except to fix clearly factual mistakes.
- Start each entry with one line in this form:
  `` `YYYY-MM-DD`: `<commit>` @ `<branch>` by `<author>` ``
- Under each entry keep exactly these sections in order:
  `## Goal`
  short statement of the intended task or checkpoint scope
  `## Changes summary`
  flat bullets describing committed changes first, then important staged/unstaged/untracked changes if they are relevant to continuing the task
  `## Verified`
  flat bullets with commands actually run and the important observed result
  `## Open items`
  flat bullets for remaining risks, missing verification, known breakage, or next recommended step
- Record only repo-relevant facts that help continuation. Skip conversational history, speculation, and incidental noise.
- When the worktree is dirty, explicitly distinguish committed, staged, unstaged, and untracked changes, but mention only files relevant to the tracked task.
- Prefer clickable file links for important files mentioned in `STATUS.md`.
- If verification was partial, say so plainly. Do not imply a full test pass when only compile checks, a single testcase, or CLI smoke checks were run.
- If a run failed, record the failing command and the actionable failure mode instead of hiding it.
- Before finishing a task that changed the practical project state, review whether `STATUS.md` still matches the actual branch/worktree state and update it if needed.

## Coding rules
- Best code, is no code!
- prefere functional style with poor functions;
  idealy do not change objects after construction, all methods do calculations 
  only reading the data in the class
- prefere high level code: numpy, pandas, xarray instead loops and native python sturctures (lists, dicts)
- use logging
- Use logging for debug outputs.
- use pathlib
- use attrs for dataclasses
- use attrs staticmethod/classmethod technique to construct from other data then is stored in the dataclass  
- Be defensive, with strong checks, but only for the user input data.
  That means error inputs must raise early. Therefore only check for existing keys in input dicts
  if these will be required down in a long calculation. Otherwise just let KeyError do the job.
- do not use "guess" default values, only obvious defaults
- Do not use other config keys in the case of a KeyError, just throw early.
- Do just basic asserts for consistency for function inputs.
- Can add more asserts if needed during debugging.
- NEVER resolve test errors by try blocks
- NEVER add runtime fallbacks or import shims to compensate for a broken or incomplete environment.
  If a declared dependency or tool is missing, report the environment problem plainly and fix the environment or tests around it, but do not implement code workarounds.
- NEVER write "self explanatory" into comments
- in comments indicate by ?? if you are not certain about intent of particular variable, function, parameter ...
- HLAVO is computational SW, basically input -> output function, input and output names are part of the function definition == code
  all input filenames are relative to the main config yaml file, mostly fixed names or derived from the model_name
  all output files are under workdir (passed to the hlavo script) and have fixed (or code given) names.
  Exceptions only with explicitly documented reason.

## How to verify your changes

- Use `tests/run` script to run pytest as it redirects merged stdin + stderr into `pytest.log`
  otherwise you may mis the actual output, which breaks your feedback loop. 
