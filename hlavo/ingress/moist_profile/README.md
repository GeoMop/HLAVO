# HLAVO Data aquisition


## Data scraping
- `dataflow_grab.py` - automatically process webpage of DataFlow (xpert.nz) and downloads data reports from Oddysey Xtreem

- `process_data.py` - reads various data from meteo station (meteo CSV, PR2, Oddysey), filter, cut and plot data
- `process_data_lab.py` - reads various data from laboratory (atm, flow, PR2, Oddysey), filter, cut and plot data

Supposes data in dir structure:
    
    - hlavo_data
        - data_lab
        - data_station

## Data processing into zarr_fuse storage

input files:
- `profile_schema.yaml` - zarr_fuse schema for profile network data
- `20260301T224908_dataflow_grab` - [dvc] DataFlow data for 01-06 months of 2025
- `20260301T225923_dataflow_grab` - [dvc] DataFlow data for 07-12 months of 2025
- `extract/site_coords.csv` - [GDrive] time dependent list of measuring sites with coordinates
  (updated manually from: https://docs.google.com/spreadsheets/d/104KK98NgPSF4YyyjK7BcuFe-79jfCNF_/)
- `extract/site_status.csv` - [GDrive] time dependent list of sites status
  (updated manually from: https://docs.google.com/spreadsheets/d/104KK98NgPSF4YyyjK7BcuFe-79jfCNF_/)

data processing:
- `extract/main.py`
- `extract/profile_extract.py` - includes `extract_df()` which processes a single DataFlow (xpert.nz) CSV file
  and prepares a dataframe following the schema

auxiliary files:
- `extract/export.gpx` - mapy.cz gpx file of sites locations (for extracting elevations)
- `extract/gpx2csv` - converts `export.gpx` into CSV
