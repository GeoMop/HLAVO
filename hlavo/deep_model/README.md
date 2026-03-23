# Creation of MODFLOW Based Deep Model

State on 11.2.2026

## Main workflow

1. Build grid/layers/material IDs from GIS and write base MODFLOW input files (infrequent):
```
python build_modflow_grid.py --config <config_file>
```
2. Add material parameters from YAML (frequent tuning/calibration):
```
python add_material_parameters.py --config <config_file>
```
3. Run MODFLOW:
```
python run_model.py --config <config_file>
```
4. Create Paraview and plot visualizations:
```
python visualize_results.py --config <config_file>
```

## Scenario batch workflow

For repeated scenario runs of one fixed model geometry (for example `with_mine_fine`),
use:

```
python run_scenarios.py \
  --base-config config/base/with_mine_fine.yaml \
  --scenarios-dir config/scenarios/with_mine_fine
```

Scenario files are YAML overlays over the base config.
Use `scenario.name` and `scenario.description` for metadata.

Each run gets an immutable folder:

- `runs/<model_name>/<timestamp>_<scenario_name>_<hash>/config_resolved.yaml`
- `runs/<model_name>/<timestamp>_<scenario_name>_<hash>/logs/workflow.log`
- `runs/<model_name>/<timestamp>_<scenario_name>_<hash>/workspace/<model_name>/...` outputs

Batch summary index:

- `runs/<model_name>/index.csv`

The index stores scenario name, status, duration, config path, git commit, key forcing/material values,
and simple output metrics (head statistics and groundwater-surface change statistics).

Notes:

- Grid is reused by default across scenarios when geometry inputs are identical
  (`qgis_project_path`, boundary, raster group, `meshsteps`).
- Use `--no-reuse-grid` to force rebuilding grid for each scenario.
- Use `--skip-grid` only if each run workspace already contains a valid `grid_materials.npz`.

`run_model.py` and `visualize_results.py` both expect `grid_output_path` and
`material_parameters_output_path` to exist (produced by steps 1 and 2).
The plot outputs now include a groundwater-surface elevation map.
Cross-sections can be zoomed to the upper part via `plots.xsection_depth_window`.
Temporal visualization includes:
- groundwater change map (`groundwater_change_name`)
- groundwater hydrograph at selected cell (`hydrograph_name`)
- X/Y section groundwater profiles for initial/mid/final times (`xsection_*_times_name`)
- material class X/Y cross-sections (`material_class_x_section.png`, `material_class_y_section.png`)
- ParaView cell data `material_class` (0=other, 1=sand, 2=clay)

For long-term setup with visible level changes, use `config/model_longterm.yaml`.

Material parameters in config are grouped by `materials`:
- `materials.all` is required and contains defaults plus all former `unsat` parameters.
- `materials.sand` and `materials.clay` define only `horizontal_conductivity`, `vertical_conductivity`.
- automatic assignment rule for interfaces starting with `Q`:
  - `Q*_base` -> sand
  - `Q*_top` -> clay
  - non-`Q` interfaces use `materials.all` defaults



## structure

- `GIS/`: GIS project with resources for Uhelná locality
- `model/`: workdir prepared for potential MODFLOW input/output files produced by model preparation and model run scripts

- `qgis_reader.py` - module to read and resample data from GIS resources, produces numpy arrays
- `build_model_grid.py` - uses surface map arryas produced by qgis_reader to constructe modflow geometry with assigned materials
- `model_config.yaml` - basic model configuration



## Paraview states
- `surfaces_blocks.(py | pvsm)` - paraview vizuaization of the surfaces produced by `qgis_reader.py:write_vtk_surfaces()`



### Codex files
- `AGENTS.md`: meta-instructions for how CODEX should operate; project specifics and coding style
- `PLAN.md`: project-specific, describing the general plan
-




- `.codex/config.toml`: deterministic CODEX configuration; we allow dangerous operations because we run inside a Docker sandbox
- `docker/`: Docker container setup with:
  - CODEX support
  - QGIS and PyQGIS installed, but we ultimately **do not use them** (segfaults, and they require read-write access)
  - MODFLOW 6 installed
- `src/`: CODEX working directory
  - `tests/`
    - `run`: script to run tests; a wrapper around `pytest` that captures `stdout`/`stderr`. CODEX must run tests through this wrapper, otherwise it cannot capture output when tests fail.

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
