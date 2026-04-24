from pathlib import Path

import pandas as pd
import polars as pl

from hlavo.misc.aux_zarr_fuse import (
read_storage,
remove_storage,
)

root_path = Path(__file__).parents[2]
SCHEMAS_PATH = root_path/ "hlavo/schemas"

"""
Helper script for fast reading and inspecting the created zarr_fuse storages.
Most of the storages created in HLAVO are commented out - use them.
"""

def main():
    # node, ds = read_storage(schema_path=SCHEMAS_PATH / "hlavo_surface_schema.yaml",
    #                         node_path=['chmi_aladin_10m'],
    #                         var_names=["air_pressure_at_sea_level"],
    #                         storage_path=root_path / "tests/ingress/scrapper/test_meteo_storage")
    # node, ds = read_storage(schema_path=SCHEMAS_PATH / "profile_schema.yaml",
    #                         node_path=['Uhelna', 'profiles'],
    #                         var_names=["moisture", "probe_id", "sensor_depth"],
    #                         storage_path=root_path / "tests/ingress/moist_profile/test_storage/")
    # node, ds = read_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml",
    #                         node_path=['chmi_stations'],
    #                         var_names=[],
    #                         storage_path=root_path / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_storage")
    # node, ds = read_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml",
    #                         node_path=['Uhelna', 'parflow', 'version_01'],
    #                         var_names=[],
    #                         storage_path=root_path / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_storage")
    node, ds = read_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml",
                            node_path=['Uhelna', 'parflow', 'version_01'],
                            var_names=[])
    print(ds)

    # remove_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml")

if __name__ == "__main__":
    main()
