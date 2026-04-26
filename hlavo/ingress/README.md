# Ingress scripts and configuration

This directory contains the data-ingress code that updates the zarr-fuse
datasets exposed through `hlavo/schemas`.

## Schema update procedures

- `moist_profile/profile_schema.yaml`
  - Raw source: DataFlow/Odyssey profile CSV exports plus site metadata CSVs.
  - Update scripts: `moist_profile/dataflow_grab.py`,
    `moist_profile/extract/gpx2csv.py`, and
    `moist_profile/extract/main.py`.
  - Writes `Uhelna/profiles` and `Uhelna/lab`.

- `well_data/wells_schema.yaml`
  - Raw source: Uhelná well XLSX files pulled with DVC.
  - Update script: `well_data/well_data_process.py`.
  - Writes `Uhelna/water_draw` and `Uhelna/water_levels`.

- `meteo_playground/chmi_stations/chmi_stations_schema.yaml`
  - Raw source: CHMI station metadata/data and Open-Meteo archive data.
  - Update scripts: `meta_description.py`, `meta_processing.py`,
    `data_scrapper.py`, and `data_processing.py`.
  - Writes raw station/Open-Meteo nodes and the ParFlow/CLM forcing node.

- `scrapper/schemas/hlavo_surface_schema.yaml`
  - Raw source: configured automatic service endpoints.
  - Update path: `ingress-server` worker using `scrapper/endpoints_config.yaml`
    and extractors in `scrapper/extract/`.
  - Writes forecast/surface meteorology nodes such as `yr.no` and
    `chmi_aladin_10m`.

## Subdirectories

- `meteo_playground` - scripts and examples for CHMI station, Open-Meteo, and
  ALADIN source data.
- `meteo_historical` - historical CHMI station metadata selection helpers.
- `moist_profile` - DataFlow/Odyssey moisture profile processing.
- `scrapper` - automatic ingress-service endpoint configuration and extractors.
- `well_data` - water-level and draw XLSX processing.
