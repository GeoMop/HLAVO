# HLAVO dev environments

This directory provides configurations and scripts to setup and run hlavo environment.
- core environment is based on conda
- docker wrapper image is provided for better portability and isolation
- image building script also enable conversion to SIF file for running on the (charon) cluster
- codex image provides sandboxing of the whole environmnet and codex support

## Create the conda environment (via cndenv.sh)

The script builds the conda environment according to  `conda-requirements.yml`.
It will install Miniconda and mamba in user space if needed.

  ./cndenv.sh rebuild

To run a command inside the environment:

  ./cndenv.sh run python -v

Run the smoke test

  ./cndenv.sh run python conda_env_test.py

  
## Files overview

- cndenv.sh: single entrypoint script for conda/mamba or venv setup.
- conda_env_test.py: environment smoke test (mf6 check, imports, cfgrib/eccodes).
- conda-requirements.yml: conda environment definition (name, channels, deps).
- hlavo_dockerfile: base image with Python 3.11, MF6, and requirements.
- codex_dockerfile: extends the base image with Node.js and Codex CLI.
