from pathlib import Path

import dotenv
import pandas as pd
import polars as pl
import zarr_fuse

root_path = Path(__file__).parents[2]
SCHEMAS_PATH = root_path/ "hlavo/schemas"


def load_dotenv():
    dotenv.load_dotenv(root_path / ".env")


def override_local_storage(schema, storage_path: str | Path | None):
    if storage_path is not None:
        schema.ds.ATTRS['STORE_URL'] = str(storage_path)


def load_schema(schema_path: str | Path):
    # Add .env if accessing S3
    load_dotenv()
    schema_path = Path(schema_path)
    schema = zarr_fuse.schema.deserialize(schema_path)
    return schema


def get_nested(node, path):
    for key in path:
        node = node[key]
    return node


def read_storage(schema_path: str | Path,
                 node_path: list[str], var_names: list[str],
                 storage_path: str | Path = None):
    schema = load_schema(schema_path)
    override_local_storage(schema, storage_path)

    root_node = zarr_fuse.open_store(schema)
    # rdf = root_node['Uhelna']['profiles'].read_df(var_names=["moisture", "probe_id", "sensor_depth"])
    # rdf = root_node['chmi_aladin_10m'].read_df(var_names=var_names)
    # print(rdf)
    # rdf = root_node['chmi_aladin_10m'].dataset

    node = get_nested(root_node, node_path)
    if len(var_names) > 0:
        ds = node.dataset[var_names]
    else:
        ds = node.dataset

    # place debug pause here and view pandas dataframe
    # rdf_pd = rdf.sort(["site_id", "date_time", "depth_level"]).to_pandas()
    return node, ds


def remove_storage(schema_path: str | Path, storage_path: str | Path = None):
    schema = load_schema(schema_path)
    override_local_storage(schema, storage_path)
    print(f"Removing storage at {schema.ds.ATTRS['STORE_URL']} ...")
    zarr_fuse.remove_store(schema, STORE_URL=storage_path)


def main():
    # read_storage(schema_path=SCHEMAS_PATH / "hlavo_surface_schema.yaml",
    #              node_path=['chmi_aladin_10m'],
    #              var_names=["air_pressure_at_sea_level"],
    #              storage_path=root_path / "tests/ingress/scrapper/test_meteo_storage")
    # read_storage(schema_path=SCHEMAS_PATH / "profile_schema.yaml",
    #              node_path=['Uhelna', 'profiles'],
    #              var_names=["moisture", "probe_id", "sensor_depth"],
    #              storage_path=root_path / "tests/ingress/moist_profile/test_storage/")
    # read_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml",
    #              node_path=['chmi_stations'],
    #              var_names=[],
    #              storage_path=root_path / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_storage")
    read_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml",
                 node_path=['Uhelna', 'parflow', 'version_01'],
                 var_names=[],
                 storage_path=root_path / "hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_storage")

    # remove_storage(schema_path=SCHEMAS_PATH / "chmi_stations_schema.yaml")

if __name__ == "__main__":
    main()
