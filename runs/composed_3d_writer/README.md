# Composed 1D-3D run with ZARR prediction output

Runs the composed model over March 2025 for the 1D sites 1 and 2 and writes the predictions
into a local ZARR store. It is the run counterpart of the unit test
`tests/composed/test_prediction_writer.py::test_zarr_prediction_writer_coord_sizes`.
The run needs no network access.

## Inputs
- `prepare_inputs.py` creates synthetic local stores in `inputs/` (profiles, ParFlow meteo
  forcing, wells) plus schema copies whose `STORE_URL` points to them. `run_simulate.sh` calls it
  before every simulation, inside the docker image (the schema copies contain absolute paths).
- 1D model: `Model1D` with the `SurfaceScalingMock` surface model (no ParFlow, no Kalman filter).
- 3D model: `Model3DDelay` mock (single water-level accumulator with drainage).
  The real MODFLOW backend follows after milestone 4.
- Real data: point `model_1d.schema_files` and `wells_schema_file` in `config.yaml` to the schemas in
  `hlavo/ingress/...` (S3 stores, credentials in `.secrets_env` in the repository root).

## Run
```
runs/composed_3d_writer/run_simulate.sh
```
Outputs in this folder: `simulation.zarr`, `hlavo_main.log`, `worker_1d_site_<id>.log`
(and the generated `inputs/`).

## Output
`simulation.zarr` (structure `hlavo/schemas/simulation_schema.yaml`):
- `Uhelna/site_prediction`: `velocity` (1D recharge) and `pressure_head` (3D head sent to 1D)
  per `date_time`, `site_id`, `calibration`.
- `Uhelna/well_prediction`: `water_level` per `date_time`, `well_id`, `calibration`,
  for the wells of the wells store.

`calibration` is the start datetime of the run. The writer replaces the nodes on every run
(`write_ds(mode="w")`), so the store holds only the last run.

Quick inspection:
```
dev/hlavo run python -c "import xarray as xr; print(xr.open_zarr('runs/composed_3d_writer/simulation.zarr', group='Uhelna/site_prediction'))"
```

## Known limitations
- `APCP` is declared (and generated) in mm/s, `SurfaceScalingMock` treats precipitation as mm/day,
  so the recharge is far too small and the water level stays near -60 m. See QaR in `PLAN.md`.
