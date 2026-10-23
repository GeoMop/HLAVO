# Mock of the composed model run


## 1D model
The section "sites" is just a mock, to provide some geographical locations of mock wells for the 3d model. In real composed model this should be read from GIS by 1d models and send to the 3d model through the queue.

## 3D model
Defined in the "model_3d" section of the config file 
see comments in the section.

## Geometry build (`run_build_model.sh`)
`hlavo build_model config.yaml -w <this folder>` reads the QGIS project given in
`model_3d.geometry` (GIS rasters needed, DVC) and writes into `geometry/`:
- `geometry.nc`: the model geometry (grid, terrain-following layers, geologic unit of every cell,
  unit base surfaces, boundary polygon); `Geometry.load(<folder>)` in
  `hlavo/deep_model/geometry.py` recreates it without the GIS, e.g. on a cluster;
- `geometry.vtm` (+ `geometry/`): VTK multiblock for ParaView (local XY coordinates), blocks
  `cells` (cell data `material`, `active`), `surface_<unit>`, `boundary`.
Log: `build_model.log`. No material properties and no MODFLOW files yet (milestone 5).

