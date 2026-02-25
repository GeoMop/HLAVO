# HLAVO dev environment

This directory provides configurations and scripts to setup and run hlavo environment.
- core environment is based on conda (modflow6 + python packages), parflow must be installed extra
- docker wrapper image is provided for better portability and isolation ( contains conda + parflow)
- image building script also enable conversion to SIF file for running on the (charon) cluster
- codex image provides sandboxing of the whole environmnet and codex support

## Build and run

Build or update the docker (or conda [-c]) environment. Optionally, remove [-f] and start from scratch:

```bash
./hlavo-build [-c] [-f] build
```

Run a command inside the environment. Use `-t` to enforce interactive terminal:

```bash
./hlavo [-c] [-t] [run] <cmd>
./hlavo [-c] [shell|codex]
```

`run` - execute <cmd> command with arguments within the environment
`shell` - <WIP>
`codex` - run codex within hlavo environment


Notes:
- `-c` uses CONDA environmnet to run the command, docker container is used otherwise.
- `-t` sets `TERM=-it` so Docker allocates a TTY, this is necessary only to run from other script.
- UID/GID are passed into the container to match host ownership.

Push the built image to Docker Hub (tags as `flow123d/hlavo:<tag>`):

```bash
./hlavo [-c] [-t] run <cmd>
```

Notes:
- `-c` uses CONDA environmnet to run the command, docker container is used otherwise.
- `-t` sets `TERM=-it` so Docker allocates a TTY.
- UID/GID are passed into the container to match host ownership.

Push the built image to Docker Hub (tags as `flow123d/hlavo:<tag>`):


```bash
./hlavo-build push
```

## Key files

- `hlavo_common.sh`: core logic; defines base build/run and venv overlay behavior.
- `hlavo`: run wrapper; parses subcommands and delegates to `hlavo_common.sh`.
- `hlavo-build`: build wrapper; calls base build + venv overlay.
- `hlavo_dockerfile`: Docker image definition (ParFlow + conda env).
- `hlavo-entrypoint`: entrypoint to align container UID/GID with host.
- `parflow_install.sh`: ParFlow build/install script (wget archive, apt deps).
- `parflow-ldd.sh`: helper to collect ParFlow shared lib deps in the image.
- `conda-requirements.yml`: conda environment spec.
- `test_env.py`: smoke test for environment sanity.


## Future developments

- support for shell
- codex derived docker image + dockercompose
- simplified image for computing only (no ingress and qgis)
