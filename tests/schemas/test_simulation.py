from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import polars as pl
import zarr_fuse as zf

from hlavo.misc.aux_zarr_fuse import override_local_storage


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "hlavo" / "schemas" / "simulation_schema.yaml"
STORE_PATH = Path(__file__).with_name("simulations.zarr")


def _schema_with_local_store(store_path: Path):
    schema = zf.schema.deserialize(SCHEMA_PATH)
    override_local_storage(schema, store_path)
    return schema


def _well_prediction_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date_time": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 1, 0),
            ],
            "well_id": ["BH01", "BH01"],
            "calibration": ["2026-01-01", "2026-01-01"],
            "water_level": [245.1, 245.0],
            "water_depth": [3.2, 3.3],
            "longitude": [14.95, 14.95],
            "latitude": [50.32, 50.32],
            "Z": [248.3, 248.3],
        },
        schema={
            "date_time": pl.Datetime("ns"),
            "well_id": pl.String,
            "calibration": pl.String,
            "water_level": pl.Float64,
            "water_depth": pl.Float64,
            "longitude": pl.Float64,
            "latitude": pl.Float64,
            "Z": pl.Float64,
        },
    )


def test_simulation_schema_single_calibration_local_store():
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)

    schema = _schema_with_local_store(STORE_PATH)
    node_schema = schema.groups["Uhelna"].groups["well_prediction"].ds
    node = zf.open_store(schema)["Uhelna"]["well_prediction"]

    node.update(_well_prediction_df())

    reopened_node = zf.open_store(_schema_with_local_store(STORE_PATH))["Uhelna"]["well_prediction"]
    ds = reopened_node.dataset.load()

    assert ds.sizes == {"date_time": 2, "well_id": 1, "calibration": 1}
    assert ds.coords["date_time"].attrs["df_col"] == node_schema.COORDS["date_time"].df_col
    assert ds.coords["calibration"].attrs["composed"] == ["calibration"]
    assert ds["water_level"].attrs["description"] == node_schema.VARS["water_level"].description
    np.testing.assert_array_equal(
        ds.coords["date_time"].values,
        np.array(
            ["2025-12-31T22:00:00", "2025-12-31T23:00:00"],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(
        ds.coords["calibration"].values,
        np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(ds["water_level"].values[:, 0, 0], np.array([245.1, 245.0]))
    np.testing.assert_allclose(ds["water_depth"].values[:, 0, 0], np.array([3.2, 3.3]))

    df = reopened_node.read_df(
        var_names=[
            "date_time",
            "well_id",
            "calibration",
            "water_level",
            "water_depth",
            "longitude",
            "latitude",
            "Z",
        ]
    )
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 8)
    assert df["well_id"].unique().to_list() == ["BH01"]
    assert df["water_level"].to_list() == [245.1, 245.0]
    assert df["water_depth"].to_list() == [3.2, 3.3]
    assert df["longitude"].to_list() == [14.95, 14.95]
    assert df["latitude"].to_list() == [50.32, 50.32]
    assert df["Z"].to_list() == [248.3, 248.3]
    assert df["calibration"].dt.strftime("%Y-%m-%d %H:%M:%S").unique().to_list() == [
        "2026-01-01 00:00:00"
    ]
