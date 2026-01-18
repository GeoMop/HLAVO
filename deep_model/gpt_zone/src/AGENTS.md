# AGENTS.md

Goal: Script for creating modflow6 model from GIS data.


## Project structure

This is subproject of HLAVO repository living in deep_model/gpt_zone.
Codex run within docker container

## QGIS input file structure
Lives in read only GIS directory.
Project file: `uhelna_all.qgz`

Layers:
- `JB_extended_domain` : plane view domain boundary polygon
- `HG model layers` : group with tif rasters = elevation maps for bases of geological layers
  order is from surface to bottom
  
Ignore anything other.

## High level plan to build the model

- for given meshsteps create rectangular grid covering the domain (XY plane) and vertical range of layer surfaces (Z coord)
- XY dependent mask of active columns = inside or intersected by the boundary polygon
- for layer_name, L in layer bases in bottom up order:
   - where L[x,y] > last_top[x,y] : 
       - set material layer_name for cells: last_top[x,y] <= cell_z < L[x,y] 
       - last_top[x,y] = L[x,y]
- refinement of cell Z coords to match the layer boundaries, adjust also interior layer cells for even Z steps over the single layr on single X,Y column
       
Use layers from the `HG model layers` group, the first is the top `relief` raster.
All is in Krovak coordinates. These gives enormous numbers with loose of precision we
must use local coordinates relative to a reference point - use e.g. first point of the polygon
rounded to 1000m.

## Coding rules
- Be defensive, with strong checks, only for the input data.
- Do just basic asserts for consistency for function inputs.
- Can add more asserts if needed during debugging.
- Use logging for debug outputs.
- NEVER resolve errors by try blocks

- use logging
- use pathlib
- use attrs for dataclasses
- prefere functional style with poor functions;
  idealy do not change objects after construction, all methods do calculations 
  only reading the data in the class
- use attrs staticmethod/classmethod technique to construct from other data then is stored in the dataclass  
- use a single input yaml file to parametrize the model construction
- prefere high level code: numpy, pandas, xarray instead loops and native python sturctures (lists, dicts)
       
## Modflow model
- the model creation script only creates the geometry and materials
- create any model inputs in the `model` directory, run the model in that directory producing any outputs there
- main run script:
  - set no flow on model boundaries and bottom
  - set seepage face condition on whole top boundary + 0.1mm / day rainfall 
  - set any reasonable conductivities to the layers (through input yaml) 
  - run steady Darcy flow calculation

## How to verify changes
- create a modelrun script that first call the model creation and then calls modflow for result.
- test by the run script
- fix until both model construcion and model run works
- ALWAYS run `tests/run` at the end of changes
