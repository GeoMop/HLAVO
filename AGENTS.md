# AGENTS.md

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


## `hlavo/` Project Source Structure

- `composed/` - communication of deep_model and kalman (surface model) in a parallel run
- `deep_model/` - model of the deep vadoze zone
- `ingress/` - data ingress tools and services
- `kalman/` - surface model assimilation (using profile measurements)
- `soil_parflow/` - ParFlow based surface model (Richards)
- `soil_py/` - Pure Python minimalistic Richards' solver.


## HLAVO Project Guidelines

This is subproject of HLAVO repository living in deep_model/gpt_zone.
Codex run within docker container


## Coding rules
- use logging
- use pathlib
- use attrs for dataclasses
- prefere functional style with poor functions;
  idealy do not change objects after construction, all methods do calculations 
  only reading the data in the class
- use attrs staticmethod/classmethod technique to construct from other data then is stored in the dataclass  
- prefere high level code: numpy, pandas, xarray instead loops and native python sturctures (lists, dicts)
- Be defensive, with strong checks, but only for the user input data.
- Do just basic asserts for consistency for function inputs.
- Can add more asserts if needed during debugging.
- Use logging for debug outputs.
- NEVER resolve test errors by try blocks

## How to verify your changes

- Use `tests/run` script to run pytest as it redirects merged stdin + stderr into `pytest.log`
  otherwise you may mis the actual output, which breaks your feedback loop. 
