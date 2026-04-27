# CHMI Meteostations Metadata and Scripts

This package owns `chmi_stations_schema.yaml`, exposed in the project schema
index as `hlavo/schemas/chmi_stations_schema.yaml`.

## CHMI metadata

- `meta1.json` - list of stations active and historic: WSI, station name, activity interval, GPS coords
- `meta2.json` - list of all measured quantities on all stations: [measurement period, WSI, from_date, to_date,quantity_short, quantity_long,unit, height, schedule_str]
- `meta3.json` - meteo state abbreviations
- `meta4.json` - data quality states

## Schema Update Procedure

1. Run `meta_description.py` when CHMI quantity metadata changes; it regenerates
   `quantity_definitions.json`.
2. Run `meta_processing.py` when station metadata or the selected domain changes;
   it regenerates nearby-station CSV outputs.
3. Run `data_scraper.py` to download CHMI station data and Open-Meteo archive
   data into local cache directories.
4. Run `data_processing.py` to update zarr-fuse nodes in
   `chmi_stations_schema.yaml` and rebuild the ParFlow/CLM forcing dataset.

Configuration of CHMI stations and quantities, download directories etc. can be set in `config.py`.

## Data Download

`data_scraper.py` creates or updates:

- `open_meteo_data_hourly`
- `stations_data_daily`
- `stations_data_hourly`

`data_processing.py` writes `chmi_stations.zarr` unless configured otherwise in
`config.py`.

## Processing Results

- `stations_nearby.csv` - automated stations near the Uhelna model + quantities they produce
- `stations_nearby_active.csv` - ... active nearby stations 
- `quantity_definitions.json` - dict of quantity_short -> {description: ..., unit: ... } mapping
- `data_processing.py` - digests downloaded data into zarr-fuse storage and prepares dataset for ParFlow/CLM
