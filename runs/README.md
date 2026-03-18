# Run Scripts And Configurations

This directory contains small run-specific folders with configs and helper
launchers for starting HLAVO simulations from a consistent entrypoint.

## Launchers

All three wrappers forward their arguments directly to
`../hlavo/main.py`. Choose the wrapper according to the environment you want to
use:

- `./run.sh <args>`
  Run `main.py` in the default docker environment.
- `./run_c.sh <args>`
  Run `main.py` in the conda-based environment.
- `./run_0.sh <args>`
  Run ``main.py` ` directly in the current shell. Use within CODEX and HLAVO environemnt shells.

  
Run-specific assets such as directories under `composed_mock`, `kalman`, or
`et_synthetic` are typically supplied as arguments to that main program.

## Run Folders
Individual simulation or calibration scenarios are organised into subfolders

- `column`
  Laboratory column data and subsets for Kalman or Richards-related analysis.
- `composed_mock`
  Mock setup for a composed 1D + 3D workflow, including `config.yaml`.
- `deep_model_only`
  Placeholder for runs focused on the deep vadose zone model.
- `et_synthetic`
  Older synthetic evapotranspiration test configurations.
- `kalman`
  Kalman and ParFlow-related run configuration files.

