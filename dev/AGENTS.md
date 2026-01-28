# Guidelines for CODEX env preparation

## Goal

- following scripts reuse code from cndenv.sh
- hlavo-build - script to build conda / docker / singularity environments
  - singularity is converted from docker image
  - docker is wrapper around conda environment + two stage build and copy of parflow
  - conda environment provides modflow6 and havy python packages
  
- hlavo - run script
  - provide function to run a comand in: conda,  docker or SIF environment;
  - docker deals with binding whole repo dir and correct UID, GID (basicaly mimicking singularity behavior)
  
## Implementation details
- core implemented in hlavo_common, hlavo-build and hlavo scripts are just interfaces

- bash functions base_run_docker, base_run_conda  implementing a call af a command with args within docker (binds + UID, GID) or within conda `hlavo` environment
- set `base_run` dynamicaly to either implementation according to the base env mode (currently env HLAVO_MODE; set this and other mode dependent variables (VENV_DIR)
  and functions in a single case statemnt
- venv_ensure and venv_overlay, should be the only core venv functions always running python through current `base_run` implementation
- function env_build - top level environment build including base_build (mode specific base_build_docker, base_build_conda) + venv_overlay 
- function run_cmd - top level run including sourcing of venv activate
- use base_build_* names for build functions (rename existing build helpers instead of adding wrappers)
- keep a single HLAVO_MODE case to set VENV_DIR and mode-specific functions (base_run/base_build)
  
## Guidelines
- minimise the code,
- constantly try to reduce duplicated code
- avoid state branching, no checks with workarounds, just assert and fail with proper indication of the cause.
- merge and redirect output of all commands and review properly the redirected output
- create hlavo_common.sh with functions and vars used from both hlavo and hlavo-build; use script location to access it consistently
- do not overparametrize: keep env variables minimal, prefer fixed defaults (IMAGE_NAME fixed, IMAGE_TAG defaults to latest, DOCKERFILE from repo root)
