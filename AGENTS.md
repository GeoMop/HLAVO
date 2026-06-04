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


## Workflow
- The user uses `git-cola` to review changes. Before writing files, check the repository state. 
  If files are already modified, stage the user edits and only then write the changes.
- Do not ask for confirmation before making requested changes; the user will review them in `git-cola`.
- Do not commit changes yourself, unless you are explicitely asked to do so.
- At the beginning of work, check the request against `AGENTS.md`, `PLAN.md`, and relevant repository docs.
- For any changes involving more than single function:
  1. Immediately estimate size of changes and report if you are planing for larger edits. Just to let operator to know he should wait for your plan. 
  2. For up to 3 item planes report them directly in console and ask for the feedback in the prompt.
  3. For more items, write them into PLAN.md last questions section. And ask user to replay to them there.
- Before finishing, review changes against the documentation and put open project-specific questions or inconsistencies/remarks (QaR) in the last section of `PLAN.md`.
- Regularly review the QaR section, incorporate my answers into the plan above and mark the item resolved. I will then remove resolved items.

- Treat `AGENT` notes in source comments or documentation as direct instructions or answers. Once resolved, add one short line after the note summarizing the resolution.
- Let user remove AGENT instructions and resolution notes. Do not remove them yourself unless ask for that explicitely.
- When reviewing `PLAN.md` or source comments, prefer the newest relevant `AGENT` note over older surrounding context. Do not summarize or act on stale requests if a later note narrows, replaces, or corrects them.
- For code changes, implement and run appropriate unit tests or the configured run script before finishing. For documentation-only changes, no tests are required.
- Do not touch code when the user asks to work on the plan only.
- Once you finish a step described in the PLAN.md, mark it as resolved and put there short description where I can find a test proving the completion.

### Mandatory Finish Checklist

Before the final response, always verify these workflow items explicitly:
- Every active `AGENT` note touched by the work has a following `Resolved:` line.
- `PLAN.md` has been reviewed, and any new open QaR or inconsistency from the work is recorded in its last section.
- Required verification commands have been run, or the final response states why they were not run.
- The final response mentions any missed requirement, open QaR, or failed verification.



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
Include: `python_coding.md`

- in comments indicate by ?? if you are not certain about intent of particular variable, function, parameter ...
- HLAVO is production software, but with internall use only for now and just single deplyment. 
  It also has the simple input -> output structure. We want basic checks of the input, good logging for inspection of long computations.
  Good documentation since it is rather complex.
- all input filenames are relative to the main config yaml file, mostly fixed names or derived from the model_name
  all output files are under workdir (passed to the hlavo script) and have fixed (or code given) names.
- Exceptions only with reason explicitly documented in the code.

## How to verify your changes

- Use `tests/run` script to run pytest as it redirects merged stdin + stderr into `pytest.log`
  otherwise you may mis the actual output, which breaks your feedback loop. 
