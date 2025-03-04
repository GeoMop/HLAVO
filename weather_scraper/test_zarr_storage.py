import json
import numpy as np
from pathlib import Path
import polars as pl
import xarray as xr
import pytest


from zarr_storage import create, update, read

@pytest.fixture
def sample_df():
    # Create a sample Polars DataFrame with multi-index columns: time_stamp and location,
    # plus two data columns: var1 and var2.
    location = [(1.,2.), (3., 4.0), (5., 6.0), (1., 3.)]
    loc_x, loc_y = zip(*location)
    loc_struct = pl.DataFrame(dict(x=loc_x, y=loc_y)).to_struct()
    df = pl.DataFrame({
        "time_stamp": [1, 2, 3, 4],
        "location": loc_struct,
        "sensor": ["A", "B", "C", "A"],
        "var1": [0.0, 0.0, 0.0, 0.0],
        "var2": [0, 0, 0, 0]
    })
    return df


def test_create(tmp_path, sample_df):
    # Create a Zarr store with index columns ["time_stamp", "location"].
    # For a dynamic time dimension, we reserve 0; for location, reserve 40 slots.
    zarr_path = tmp_path / "test.zarr"
    ds = create(zarr_path, sample_df,
                index_cols={"time_stamp":0, "location":40, "sensor": None},)
    # Check that the time coordinate is empty and location coordinate has length 40.
    assert len(ds.dims) == 3
    assert len(ds.coords) == 4
    assert ds.coords["time_stamp"].size == 0
    assert ds.coords["location_x"].size == 40
    assert ds.coords["location_y"].size == 40
    assert ds.coords["sensor"].size == 3



def test_update_and_read(tmp_path, sample_df):
    # First, create the empty store.
    zarr_path = tmp_path / "test.zarr"
    ds = create(zarr_path, sample_df, index_cols=["time_stamp", "location"], idx_ranges=[0, 40])

    # Now prepare an update: add rows with new time stamps and some location codes.
    # For example, add two rows for time 10 and one row for time 20.
    df_update = pl.DataFrame({
        "time_stamp": [10, 10, 20],
        "location": ["A", "D", "A"],
        "var1": [1.1, 2.2, 3.3],
        "var2": [100, 200, 300]
    })
    update(zarr_path, df_update)

    # Read back data between time 5 and 25 for locations A and D.
    df_read = read(zarr_path, time_stamp_slice=(5, 25), locations=["A", "D"])
    # For consistency sort the result.
    df_read = df_read.sort(["time_stamp", "location"])

    # We expect three rows:
    #   time 10, location A: var1=1.1, var2=100
    #   time 10, location D: var1=2.2, var2=200
    #   time 20, location A: var1=3.3, var2=300
    assert df_read.height == 3

    row_A10 = df_read.filter((pl.col("time_stamp") == 10) & (pl.col("location") == "A"))
    row_D10 = df_read.filter((pl.col("time_stamp") == 10) & (pl.col("location") == "D"))
    row_A20 = df_read.filter((pl.col("time_stamp") == 20) & (pl.col("location") == "A"))

    assert row_A10["var1"][0] == 1.1
    assert row_A10["var2"][0] == 100
    assert row_D10["var1"][0] == 2.2
    assert row_D10["var2"][0] == 200
    assert row_A20["var1"][0] == 3.3
    assert row_A20["var2"][0] == 300
