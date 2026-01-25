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
  - common function to install Venv with editable hlavo 
  - in the case of conda env; and if hlavo image exists, try to copy parflow from the image to venv/_parflow to make it available out of docker

## Guidelines
- minimise the code,
- constantly try to reduce duplicated code
- avoid state branching, no checks with workarounds, just assert and fail with proper indication of the cause.
- merge and redirect output of all commands and review properly the redirected output

