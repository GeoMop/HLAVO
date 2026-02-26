# HLAVO dev environment

This directory provides configurations and scripts to setup and run hlavo environment.
- core environment is based on conda (modflow6 + python packages), parflow must be installed extra
- docker wrapper image is provided for better portability and isolation ( contains conda + parflow)
- image building script also enable conversion to SIF file for running on the (charon) cluster
- codex image provides sandboxing of the whole environmnet and codex support

Environment is composed of up to three layers:
- docker - complete containerisation, optional, default on; provides compiled parflow
- conda - modflow6 + heavy python packages
- venv

## Build

### Basic Docker / Conda build
Build or update the docker (or conda [-c]) environment including the VENV overlay stored in `venv-docker` or `venv-docker` respectively.
Optionally, remove [-f] and start from scratch. Docker tag taken automaticaly from the HLAVO package version.

```bash
./hlavo-build [-c] [-f] build
```

### Build overlay only

```bash
./hlavo-build [-c] venv
```

### Push and pull docker image

```bash
./hlavo-build (push | pull )
```
Push or pull the built docker image to/from DockerHub (as `flow123d/hlavo:<tag>`)

## Run using `hlavo` script


```bash
./hlavo [-c] [-t] run <cmd>
```

Run a command inside the environment. Use `-t` to enforce interactive terminal.
HLAVO root directory is mounted as `/home/hlavo/workspace`. The `hlavo` user has the same UID as 
current user on the host, so the ownership should be set correctly.

Notes:
- `-c` uses CONDA environmnet to run the command, docker container is used otherwise.
- `-t` sets `TERM=-it` so Docker allocates a TTY.
- UID/GID are passed into the container to match host ownership.


## Use Codex within environment

```bash
./hlavo codex <codex_options>
```

Starts ChatGPT CODEX cli within the docker container. For the first time you have to 
connect to an ChatGPT account. Use the second option with verification code.
The directory `dev/.codex-docker` is used for codex configuration. `dev/.codex-docker/config.toml`
is setup with maximal permissions.

```bash
./halvo -c condex <codex_options>
```

Starts codex cli wihtin the conda environment. No separation, codex must be installed on the host. 
Default `~/.codex` settings are in use.



## Key files

- `hlavo_common.sh`: core logic; defines base build/run and venv overlay behavior.
- `hlavo`: run wrapper; parses subcommands and delegates to `hlavo_common.sh`.
- `hlavo-build`: build wrapper; calls base build + venv overlay.
- `hlavo_dockerfile`: Docker image definition (ParFlow + conda env).
- `hlavo-entrypoint`: entrypoint to align container UID/GID with host.
- `codex_dockerfile`: Overlay image adding codex install based on `flow123d/halavo` image.
- `parflow_install.sh`: ParFlow build/install script (wget archive, apt deps).
- `parflow-ldd.sh`: helper to collect ParFlow shared lib deps in the image.
- `conda-requirements.yml`: conda environment spec.
- `test_env.py`: smoke test for environment sanity.


## Future developments

- support for shell
- codex derived docker image + dockercompose
- simplified image for computing only (no ingress and qgis)
