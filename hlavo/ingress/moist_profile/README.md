# Moisture Profile Ingress

This package updates storage of `profile_schema.yaml`.

## Data Acquisition

- `profile_scraper.py` - logs into DataFlow/xpert.nz and downloads Odyssey Xtreem
  CSV reports into <HLAVO root>/hlavo_data/*_dataflow_grab.csv.
  TODO: use more specific output dir 
- `process_data.py` - reads moisture profile network data (meteo CSV, PR2, Oddysey), filter, cut and plot data.
- `process_data_lab.py` - reads laboratory data (atm, flow, PR2, Oddysey), filter, cut and plot data.

The exploratory scripts expect this local data layout: 
    - hlavo_data
        - data_lab
        - data_station

## Schema Update Procedure

1. Run `hlavo.ingress.profiles_scrape` to scrape new DataFlow CSV exports, or pull existing
   DVC-managed export directories.
   Manual selection of date interval, probes groups and scraper configuration is required inplace.
2. If changed, update `extract/site_coords.csv` and `extract/site_status.csv` from the
   project spreadsheet.
3. Run `extract/gpx2csv.py` only when `extract/export.gpx` was refreshed and
   waypoint support data must be regenerated.
4. Run `hlavo.ingress.profiles_process` to process the raw data and update the zarr-fuse nodes in
   `profile_schema.yaml`.
   TODO: describe if this includes configured filtering or manual processing through process_data.py is necesary first. 
   document data update procedure.


## Files:

- `profile_schema.yaml` - zarr-fuse schema for the profile monitoring network.
- `20260301T224908_dataflow_grab` - DVC-managed DataFlow data for 2025-01
  through 2025-06.
- `20260301T225923_dataflow_grab` - DVC-managed DataFlow data for 2025-07
  through 2025-12.
- `extract/site_coords.csv` - [GDrive] time dependent list of measuring sites with coordinates
  (updated manually from: https://docs.google.com/spreadsheets/d/104KK98NgPSF4YyyjK7BcuFe-79jfCNF_/)
- `extract/site_status.csv` - [GDrive] time dependent list of sites status
  (updated manually from: https://docs.google.com/spreadsheets/d/104KK98NgPSF4YyyjK7BcuFe-79jfCNF_/)

Implementation notes:

- `extract/profile_extract.py` contains `extract_df()`, which processes one
  DataFlow CSV file into the schema-shaped dataframe.
- `extract/profile_process.py` loads site coordinates/status, assigns probes to sites,
  splits lab/profile rows, opens the zarr-fuse store, and updates the nodes.

Auxiliary files:

- `extract/export.gpx` - mapy.cz gpx file of sites locations (for extracting elevations)
- `extract/gpx2csv.py` - converts `export.gpx` into CSV
