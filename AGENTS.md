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
- Do just basic asserts for consistency for function inputs.
- Can add more asserts if needed during debugging.
- NEVER resolve test errors by try blocks


## How to verify your changes

- Use `tests/run` script to run pytest as it redirects merged stdin + stderr into `pytest.log`
  otherwise you may mis the actual output, which breaks your feedback loop. 
