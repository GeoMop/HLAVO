from pathlib import Path

"""
Helper file contains constants including:
- directory path for downloaded and processed data
- schema node paths
- web sources links
- CHMI and Open Meteo helper constants (QoI names)
- physics constants
"""

SCRIPT_DIR = Path(__file__).resolve().parent

CHMI_STATIONS_SCHEMA_PATH = SCRIPT_DIR / "chmi_stations_schema.yaml"
CHMI_STATIONS_STORAGE_PATH = SCRIPT_DIR / "chmi_stations.zarr"
# CHMI_STATIONS_STORAGE_PATH = None

ACTIVE_STATIONS_CSV_PATH = SCRIPT_DIR / "stations_nearby_active.csv"
STATIONS_NEARBY_CSV_PATH = SCRIPT_DIR / "stations_nearby.csv"
QUANTITY_DEFINITIONS_PATH = SCRIPT_DIR / "quantity_definitions.json"

STATIONS_DATA_DAILY_PATH = SCRIPT_DIR / "stations_data_daily"
STATIONS_DATA_HOURLY_PATH = SCRIPT_DIR / "stations_data_hourly"
OPEN_METEO_DATA_HOURLY_PATH = SCRIPT_DIR / "open_meteo_data_hourly"

PARFLOW_OPEN_METEO_COMPARISON_PLOT_PATH = SCRIPT_DIR / "parflow_clm_vs_open_meteo.pdf"
SITE_COORDS_CSV_PATH = SCRIPT_DIR.parents[1] / "moist_profile" / "extract" / "site_coords.csv"

NODE_OPEN_METEO = ["Uhelna", "raw_data", "open_meteo"]
NODE_CHMI_STATIONS = ["Uhelna", "raw_data", "chmi_stations"]
NODE_PARFLOW = ["Uhelna", "parflow", "version_01"]

CLM_STATION_PRIORITY = [
    "0-203-0-20407036001",
    "0-203-0-11601",
    "0-20000-0-11603",
]
CLM_REQUIRED_SOURCE_VARS = ["SRA1H", "T", "P", "P1H", "F", "D10", "H"]
OPEN_METEO_REQUIRED_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "dlwr_estimate",
]

STATION_DF_METADATA_COLUMNS = {"STATION", "VTYPE", "date_time", "latitude", "longitude"}

HISTORICAL_DAILY_URL = "https://opendata.chmi.cz/meteorology/climate/historical/data/daily/"
HISTORICAL_HOURLY_URL = "https://opendata.chmi.cz/meteorology/climate/historical/data/1hour/"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_HOUR = 60 * 60
GRAVITY_M_S2 = 9.80665
DRY_AIR_GAS_CONSTANT = 287.05
STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8
