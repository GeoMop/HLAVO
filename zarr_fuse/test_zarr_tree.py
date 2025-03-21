import pytest
import xarray as xr
import numpy as np
import zarr
from pathlib import Path
script_dir = Path(__file__).parent
workdir = script_dir / "workdir"

def test_list_subgroups(tmp_path: Path):
    # Define the path to the Zarr store (a directory under the temporary path)
    store_path = workdir / "store.zarr"

    # Create an empty Zarr store by writing an empty Dataset.
    # Use consolidated=False so that metadata is written per group.
    empty_ds = xr.Dataset()
    empty_ds.to_zarr(str(store_path), mode="w")

    # Create a simple dataset for the child node.
    ds_child = xr.Dataset(
        {"temperature": (("time",), np.array([280.0]))},
        coords={"time": np.array([1000])}
    )
    # Write the dataset to the subgroup "child_1" using append mode.
    ds_child.to_zarr(str(store_path), group="child_1", mode="a")

    # Open the root group of the store in read mode.
    root_group = zarr.open_group(str(store_path), mode="r")

    # List the available subgroups under the root.
    child_keys = list(root_group.group_keys())

    # Debug: print(child_keys)  # Uncomment to see the output during development.

    # Verify that the subgroup "child_1" is present.
    assert "child_1" in child_keys
