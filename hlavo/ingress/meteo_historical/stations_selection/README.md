# Historical CHMI Meteostation Selection

This directory contains historical station-selection helpers. It does not update
a schema under `hlavo/schemas`; the active CHMI station schema update procedure
is documented in `../../meteo_playground/chmi_stations/README.md`.

## CHMI Metadata

- `meta1.json` - list of stations active and historic: WSI, station name, activity interval, GPS coords
- `meta2.json` - list of all measured quantities on all stations: [measurement period, WSI, from_date, to_date,quantity_short, quantity_long,unit, height, schedule_str]
- `meta3.json` - meteo state abreviations
- `meta4.json` - data quality states

## Metadata Processing Scripts

- `meta_description.py` - produce `quantity_definitions.json`
- `meta_processing.py` - list nearby stations

## Processing Results

- `stations_nearby.csv` - automated stations near the Uhelna model + quantities they produce
- `stations_nearby_active.csv` - ... active nearby stations 
- `quantity_definitions.json` - dict of quantity_short -> {description: ..., unit: ... } mapping
