# Automatic data ingress

This package owns `schemas/hlavo_surface_schema.yaml`, exposed in the project
schema index as `hlavo/schemas/hlavo_surface_schema.yaml`.

## Schema `schemas/hlavo_surface_schema.yaml`

Zarr-fuse schema for automatically collected meteorological data covering the
configured Uhelná geographical domain.

Unlike the other schemas in `hlavo/schemas`, this schema is updated through the
automatic `ingress-server` worker rather than a one-shot local processing
script.

## Update Procedure

1. Configure active endpoints in `endpoints_config.yaml`.
2. The ingress worker downloads each configured payload into its queue.
3. The worker imports the configured extractor module and function from
   `extract/`.
4. The extractor returns a Polars dataframe or xarray dataset matching the
   target node in `schemas/hlavo_surface_schema.yaml`.
5. The ingress worker writes the target zarr-fuse node.

Current endpoints:

- `hlavo-surface-forecast` downloads yr.no JSON forecasts and normalizes them
  with `extract/hlavo_extract.py::normalize` into the `yr.no` node.
- `chmi-aladin-1km` downloads CHMI ALADIN 1 km GRIB files and transforms them
  with `extract/chmi_aladin_1km_extract.py::extractor` into the
  `chmi_aladin_10m` node.

## Tests

Use the project test wrapper from `tests/`:

```bash
PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH \
PYTEST_ADDOPTS="ingress/scrapper/test_process_one_meteo_raw.py" \
bash ./run
```

## yr.no

The yr.no endpoint uses the Met Norway Locationforecast API. The input grid is
configured in `endpoints_config.yaml` through
`dataframes/hlavo_surface_dataframe.csv`, and normalized records include time,
location, grid flag, and available forecast quantities.
