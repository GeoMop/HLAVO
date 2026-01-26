# Environment for Developing a MODFLOW Model with CODEX

## Directory structure

- `../GIS`: GIS project, **read-only**
- `.codex/config.toml`: deterministic CODEX configuration; we allow dangerous operations because we run inside a Docker sandbox
- `docker/`: Docker container setup with:
  - CODEX support
  - QGIS and PyQGIS installed, but we ultimately **do not use them** (segfaults, and they require read-write access)
  - MODFLOW 6 installed
- `src/`: CODEX working directory
  - `AGENTS.md`: meta-instructions for how CODEX should operate; project specifics and coding style
  - `PLAN.md`: project-specific, describing the general plan
  - `tests/`
    - `run`: script to run tests; a wrapper around `pytest` that captures `stdout`/`stderr`. CODEX must run tests through this wrapper, otherwise it cannot capture output when tests fail.
- `model/`: prepared for potential MODFLOW input/output files produced by model preparation and model run scripts

## CODEX workflow

- Run the CODEX CLI inside the sandbox:  
  `./codex.sh`
- Verify that you have a clean Git state.
- Review code/tests and choose the next compact change, such as:
  - implement a small class with a few methods
  - implement a specific non-trivial function
  - perform a consistent refactor across the codebase
- Instruct CODEX CLI (you can use `Ctrl-J` for a newline; `Enter` = send).
- You can add instructions while it is processing.
- Review changes using a suitable Git client (`git-cola` suggested).
- Run the tests yourself and review test output. 
  Start bash in docker:
  `./codex.sh bash`
- Iterate until you are satisfied with the step you want to achieve.

## Experience

- Make sure CODEX has all the access it needs, including access to scripts it runs. Otherwise, it may attempt mysterious workarounds.
- Make sure it has access to test output; use the wrapper.
- Start with small steps.
- Never commit without code review.
- While CODEX may be capable of larger, pull-request-sized changes with a nuanced `AGENTS.md`,
  the reliability of the current setup is not sufficient for changes that large.
