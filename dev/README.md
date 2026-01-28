# HLAVO dev environment

This directory provides configurations and scripts to setup and run hlavo environment.
- core environment is based on conda (modflow6 + python packages), parflow must be installed extra
- docker wrapper image is provided for better portability and isolation ( contains conda + parflow)
- image building script also enable conversion to SIF file for running on the (charon) cluster
- codex image provides sandboxing of the whole environmnet and codex support

## 1) Docker: build and run

Build or update the docker (or conda [-c]) environment. Optionally, remove [-f] and start from scratch:

```bash
./hlavo-build [-c] [-f] build
```

Run a command inside the environment. Use `-t` to enforce interactive terminal:

```bash
./hlavo [-t] run <cmd>
```

Notes:
- `-t` sets `TERM=-it` so Docker allocates a TTY.
- UID/GID are passed into the container to match host ownership.


## 3) Key files

- `hlavo_common.sh`: core logic; defines base build/run and venv overlay behavior.
- `hlavo`: run wrapper; parses subcommands and delegates to `hlavo_common.sh`.
- `hlavo-build`: build wrapper; calls base build + venv overlay.
- `hlavo_dockerfile`: Docker image definition (ParFlow + conda env).
- `hlavo-entrypoint`: entrypoint to align container UID/GID with host.
- `conda-requirements.yml`: conda environment spec.
- `test_env.py`: smoke test for environment sanity.
