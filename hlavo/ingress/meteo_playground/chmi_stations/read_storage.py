from pathlib import Path
from dotenv import load_dotenv
import polars as pl
import zarr_fuse

import pandas as pd

root_path = Path(__file__).parents[4]
SCHEMA_PATH = root_path/ "hlavo/schemas/hlavo_surface_schema.yaml"


def override_local_storage(schema, storage_path: str | Path | None):
    if storage_path is not None:
        schema.ds.ATTRS['STORE_URL'] = str(storage_path)


def load_schema():
    # Add .env if accessing S3
    # load_dotenv("../.env")
    schema_path = Path(SCHEMA_PATH)
    schema = zarr_fuse.schema.deserialize(schema_path)
    return schema


def read_storage(storage_path: str | Path = None):
    schema = load_schema()
    override_local_storage(schema, storage_path)

    root_node = zarr_fuse.open_store(schema)
    rdf = root_node['chmi_aladin_10m'].read_df(var_names=["moisture", "probe_id", "sensor_depth"])
    print(rdf)

    # place debug pause here and view pandas dataframe
    rdf_pd = rdf.sort(["site_id", "date_time", "depth_level"]).to_pandas()
    pass

# -----------------------------
# main: call subscripts
# -----------------------------
def main():
    read_storage(root_path / "tests/ingress/scrapper/test_meteo_storage")


if __name__ == "__main__":
    main()
