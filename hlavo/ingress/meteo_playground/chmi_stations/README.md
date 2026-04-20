# CHMI meteostations metadata and scripts 

## CHMI metadata

- `meta1.json` - list of stations active and historic: WSI, station name, activity interval, GPS coords
- `meta2.json` - list of all measured quantities on all stations: [measurement period, WSI, from_date, to_date,quantity_short, quantity_long,unit, height, schedule_str]
- `meta3.json` - meteo state abreviations
- `meta4.json` - data quality states

## metadata processing scripts
- `meta_description.py` - produce `quantity_definitions.json`
- `meta_processing.py` - list nearby stations

## data download

- `data_scrapper.py` - download CHMI data and Open Meteo data

creates directories:

    chmi_stations.zarr
    open_meteo_data_hourly
    stations_data_daily
    stations_data_hourly

## processing results
- `stations_nearby.csv` - automated stations near the Uhelna model + quantities they produce
- `stations_nearby_active.csv` - ... active nearby stations 
- `quantity_definitions.json` - dict of quantity_short -> {description: ..., unit: ... } mapping
- `data_processing.py` - digest downloaded data into zarr_fuse storage and prepares dataset for Parflow/CLM
