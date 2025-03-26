import json
import yaml
import numpy as np
import numpy.testing as npt
from pathlib import Path
import polars as pl
import xarray as xr
import pytest
import zarr

script_dir = Path(__file__).parent
inputs_dir = script_dir / "inputs"
workdir = script_dir / "workdir"

from zarr_storage import Node
from zarr_structure import read_structure




@pytest.mark.skip
def test_update_xarray_nd():
    # Create an initial dataset with dimensions: time, loc, sensor.
    # For fixed dims "loc" and "sensor", we allocate extra space.
    ds = xr.Dataset(
        data_vars={"var_1": (("time", "loc", "sensor"),
                           np.full((2, 3, 2), 100.0, dtype=float)),
                   "var_2": (("time", "loc", "sensor"),
                    np.full((2, 3, 2), 100.0, dtype=float))
                   },
        coords={
            "time": np.array([1, 2]),
            "loc": np.array(["C", "A", "_B"]),  # allocated size=3, initially valid size=2
            "sensor": np.array(["Y", "X"])  # allocated size=2, valid size=2
        }
    )
    ds.attrs["append_dim"] = "time"
    ds.attrs["coords_valid_size"] = {"time": 2, "loc": 2, "sensor": 2}

    # Create an update DataFrame.
    # For time=2, update: (loc="A", sensor="X") -> 200.0 and (loc="D", sensor="X") -> 210.0.
    # Note: "D" is a new fixed value for "loc" (should be appended to valid region if space permits).
    # For time=3 (new dynamic), update: (loc="B", sensor="Y") -> 300.0.
    df = pl.DataFrame({
        "time": [2, 2, 2,  3, 3, 3,  3, 3, 3],
        "loc": ["C", "A", "B", "C", "A", "B", "C", "A", "B"],
        "sensor": ["X", "X", "X", "X", "X", "X", "Y", "Y", "Y"],
        "var_1": [200.0, 210.0, 220.0,   300.0, 310.0, 320.0,   350.0, 360.0, 370.0],
        "var_2": np.array([200.0, 210.0, 220.0,   300.0, 310.0, 320.0,   350.0, 360.0, 370.0])+1000,
    })

    ds_updated = update_xarray_nd(ds, df)

    # Check that the dynamic dimension "time" now includes time=3.
    updated_times = ds_updated.coords["time"].values
    np.testing.assert_array_equal(updated_times, np.array([1, 2, 3]))

    # Check that fixed dimension "loc" now has an increased valid region including "D".
    assert ds_updated.attrs["coords_valid_size"]["loc"] == 3
    loc_valid = ds_updated.coords["loc"].values
    np.testing.assert_array_equal(loc_valid, np.array(["C", "A", "B"]))

    # Verify variable updates:
    # At time=2, (loc="A", sensor="X") should be updated to 200.0.
    var_1 = ds_updated["var_1"].to_numpy()
    ref_var_1 = np.array([
        # time = 1 (unchanged)
        [[100.0, 100.0],  # loc "A": sensor "Y", "X"
         [100.0, 100.0],  # loc "B"
         [100.0, 100.0]],  # loc "C"

        # time = 2
        [[100.0, 200.0],  # loc "A": sensor "Y" remains 100, sensor "X" updated to 210.0
         [100.0, 210.0],  # loc "B": sensor "Y" remains 100, sensor "X" updated to 220.0
         [100.0, 220.0]],  # loc "C": sensor "Y" remains 100, sensor "X" updated to 200.0

        # time = 3
        [[350.0, 300.0],  # loc "A": sensor "Y" updated to 360, sensor "X" updated to 310
         [360.0, 310.0],  # loc "B": sensor "Y" updated to 370, sensor "X" updated to 320
         [370.0, 320.0]]  # loc "C": sensor "Y" updated to 350, sensor "X" updated to 300
    ])

    np.testing.assert_almost_equal(var_1, ref_var_1)

    var_2 = ds_updated["var_2"].to_numpy()
    ref_var_2 = np.array([
        # time = 1 (unchanged)
        [[100.0, 100.0],
         [100.0, 100.0],
         [100.0, 100.0]],

        # time = 2
        [[100.0, 1200.0],  # loc "A": sensor "Y" remains 100, sensor "X" becomes 1210.0
         [100.0, 1210.0],  # loc "B": sensor "Y" remains 100, sensor "X" becomes 1220.0
         [100.0, 1220.0]],  # loc "C": sensor "Y" remains 100, sensor "X" becomes 1200.0

        # time = 3
        [[1350.0, 1300.0],  # loc "A": sensor "Y" becomes 1360, sensor "X" becomes 1310
         [1360.0, 1310.0],  # loc "B": sensor "Y" becomes 1370, sensor "X" becomes 1320
         [1370.0, 1320.0]]  # loc "C": sensor "Y" becomes 1350, sensor "X" becomes 1300
    ])
    np.testing.assert_almost_equal(var_2, ref_var_2)

    print("Test passed: update_xarray_nd with separate coordinate processing works correctly.")

"""
This is an inital test of xarray, zarr functionality that we build on.
This requires dask.
"""
def aux_read_struc(fname):
    struc_path = inputs_dir / fname
    structure = read_structure(struc_path)
    assert set(['COORDS', 'VARS']).issubset(set(structure.keys()))

    store_path = (workdir / fname).with_suffix(".zarr")
    local_store = zarr.storage.LocalStore(store_path)

    # memory_store = zarr.storage.MemoryStore()

    # zip_store = zarr.storage.ZipStore('path/to/archive.zip', mode='w')

    # s3_fs = fsspec.filesystem('s3', key='YOUR_ACCESS_KEY', secret='YOUR_SECRET_KEY')
    # s3_store = zarr.FSStore('bucket-name/path/to/zarr', filesystem=s3_fs)

    tree = Node.create_storage(structure, local_store)
    return structure, local_store, tree

@pytest.mark.skip
def test_node_tree():
    # Read the YAML file from the working directory.
    # The file "structure_tree.yaml" must exist in the current working directory.
    # Example YAML file content (as a string for illustration):
    structure, store, tree = aux_read_struc("structure_tree.yaml")

    # Create a mapping from node names to minimal Polars DataFrames.
    # Each node is updated with unique values.
    df_map = {
        "": pl.DataFrame({"time": [1000], "temperature": [280.0]}),
        "child_1": pl.DataFrame({"time": [1001], "temperature": [281.0]}),
        "child_2": pl.DataFrame({"time": [1002], "temperature": [282.0]}),
        "child_1/child_3": pl.DataFrame({"time": [1003], "temperature": [283.0]}),
    }

    # Recursively update each node with its corresponding data.
    def update_tree(node: Node):
        assert len(node.dataset.coords) == 1
        assert len(node.dataset.data_vars) == 1

        if node.group_path in df_map:
            node.update(df_map[node.group_path])
        for key, child in node.items():
            update_tree(child)

    update_tree(tree)

    # Recursively collect nodes into a dictionary for easy lookup.
    def collect_nodes(node, nodes_dict):
        nodes_dict[node.group_path] = node
        for key, child in node.items():
            collect_nodes(child, nodes_dict)
        return nodes_dict

    root_node = Node.read_store(store)
    nodes = collect_nodes(root_node, {})

    # Expected values for each node: (time coordinate, temperature variable)
    expected = {
        key: (df['time'].to_numpy(), df['temperature'].to_numpy())
        for key, df in df_map.items()
    }

    # Verify that each node’s dataset contains the expected coordinate and variable data.
    for node_name, (exp_time, exp_temp) in expected.items():
        ds = nodes[node_name].dataset
        np.testing.assert_array_equal(ds.coords["time"].values, exp_time)
        np.testing.assert_array_equal(ds["temperature"].values, exp_temp)

    # Verify the tree structure:
    # The root node should have children "child_1" and "child_2"
    assert set(root_node.children.keys()) == {"child_1", "child_2"}
    # Node "child_1" should have one child: "child_3"
    assert set(root_node.children["child_1"].children.keys()) == {"child_3"}



def test_read_structure_weather(tmp_path):
    # Example YAML file content (as a string for illustration):
    structure, store, tree = aux_read_struc("structure_weather.yaml")
    assert len(structure['COORDS']) == 2
    assert len(structure['VARS']) == 4
    print("Coordinates:")
    for coord in structure["COORDS"]:
        print(coord)
    print("\nQuantities:")
    for var in structure["VARS"]:
        print(var)

    # Create a Polars DataFrame with 6 temperature readings.
    # Two time stamps (e.g. 1000 and 2000 seconds) and three latitude values (e.g. 10.0, 20.0, 30.0).
    df = pl.DataFrame({
        "timestamp": [1000, 1000, 1000, 2000, 2000, 2000],
        "latitude": [10.0, 20.0, 20.0, 10.0, 20.0, 20.0],
        "longitude": [10.0, 10.0, 20.0, 10.0, 10.0, 20.0],
        "temp": [280.0, 281.0, 282.0, 283.0, 284.0, 285.0]
    })

    # Update the dataset atomically using the Polars DataFrame.
    updated_ds = tree.update(df)

    # Now, re-read the entire Zarr storage from scratch.
    new_tree = Node.read_store(store)
    new_ds = new_tree.dataset
    print("Updated dataset:")
    print(new_ds)

    # --- Assertions ---
    # We expect that the update function (via update_xarray_nd) will reshape the temperature data
    # into a (time, lat) array, i.e. shape (2, 3), with coordinates "time" and "lat".
    # Check the shape of the temperature variable.
    assert new_ds["temperature"].shape == (2, 3)

    # Check that the "time" coordinate was updated to [1000, 2000]
    np.testing.assert_array_equal(new_ds["time of year"].values, [1000, 2000])
    # Check that the "lat" coordinate was updated to [10.0, 20.0, 30.0]
    np.testing.assert_array_equal(new_ds["latitude"].values, [20.0, 20.0, 10.0])
    for row in df.iter_rows(named=True):
        time = row["timestamp"]
        lat = row["latitude"]
        lon = row["longitude"]
        assert new_ds["temperature"].sel({"time of year":time, "lat_lon":hash((lat, lon))}) == row["temp"]

@pytest.mark.skip
def test_read_structure_tensors(tmp_path):
    structure, store, tree = aux_read_struc("structure_tensors.yaml")
    assert len(structure['COORDS']) == 3
    assert len(structure['VARS']) == 2
    print("Coordinates:")
    for coord in structure["COORDS"]:
        print(coord)
    print("\nQuantities:")
    for var in structure["VARS"]:
        print(var)




def update_zarr_store(zarr_path: str, ds_update: xr.Dataset) -> None:
    """
    Update/append 'ds_update' into an existing Zarr store at 'zarr_path'.
    - Dims: 'time' and 'x'.
    - Possibly extends the 'time' dimension if new time coords go beyond existing range.
    - Uses region='auto' so xarray infers which slices to overwrite/append.
    """
    ds_update.to_zarr(
        zarr_path,
        mode="r+",        # read/write (store must already exist)
        region="auto"     # infer slice(s) from coords to overwrite or append
    )


@pytest.mark.parametrize(
    "init_time_len, update_time_len",
    [
        (100, 50),  # Example: initial time=100, then append 50 more
        # You can add more tuples if you want additional scenarios.
    ],
)
def test_xarray_zarr_append_and_update(tmp_path, init_time_len, update_time_len):
    """
    1. Create initial dataset with shape (time=init_time_len, x=50), chunked at (100, 100).
    2. Write Zarr *metadata only* (compute=False).
    3. Use 'update_zarr_store' to append a second dataset with shape (time=update_time_len, x=50).
       - Possibly extends the 'time' dimension.
    4. Verify final shape.
    5. Verify final data values.
    6. Assert the underlying Zarr chunk sizes.
    """

    # --------------------------
    # 1. CREATE INITIAL DATASET
    # --------------------------
    x_len = 50
    # We'll create distinct data so we can test that the final result is correct
    data_initial = np.random.rand(init_time_len, x_len)
    ds_initial = xr.Dataset(
        {
            "var": (("time", "x"), data_initial)
        },
        coords={
            "time": np.arange(init_time_len),  # 0..(init_time_len-1)
            "x": np.arange(x_len)
        }
    )

    # We want chunk sizes of (100, 100), though dimension sizes might be smaller.
    encoding = {
        "var": {
            "chunks": (100, 100)
        }
    }

    # Zarr store path
    zarr_path = str(tmp_path / "test.zarr")

    # -------------------------
    # 2. WRITE METADATA ONLY
    # -------------------------
    ds_initial.to_zarr(
        zarr_path,
        mode="w",         # create or overwrite store
        compute=False,    # metadata only (no actual data)
        encoding=encoding
    )

    # ----------------------------
    # 3. UPDATE / APPEND NEW DATA
    # ----------------------------
    # We'll append update_time_len new time steps, continuing from init_time_len.
    new_time_start = init_time_len
    new_time_stop  = init_time_len + update_time_len
    data_update = np.random.rand(update_time_len, x_len)
    ds_update = xr.Dataset(
        {
            "var": (("time", "x"), data_update)
        },
        coords={
            "time": np.arange(new_time_start, new_time_stop),
            "x": np.arange(x_len)
        }
    )

    # Call our custom update function to append/overwrite
    update_zarr_store(zarr_path, ds_update)

    # -----------------------
    # 4. VERIFY FINAL ZARR
    # -----------------------
    ds_final = xr.open_zarr(zarr_path)

    expected_time_len = init_time_len + update_time_len
    assert ds_final.dims["time"] == expected_time_len
    assert ds_final.dims["x"] == x_len

    # -----------------------
    # 5. VERIFY DATA VALUES
    # -----------------------
    # Original part should match data_initial
    final_initial_part = ds_final["var"].isel(time=slice(0, init_time_len)).values
    npt.assert_allclose(final_initial_part, data_initial)

    # Appended part should match data_update
    final_appended_part = ds_final["var"].isel(time=slice(init_time_len, expected_time_len)).values
    npt.assert_allclose(final_appended_part, data_update)

    # -----------------------
    # 6. CHECK CHUNK SIZES
    # -----------------------
    # We open the zarr group at a low level and assert the actual stored chunk sizes.
    zgroup = zarr.open_group(zarr_path, mode="r")
    zvar = zgroup["var"]  # Our variable name is "var"
    chunks = zvar.chunks

    # We asked for chunk sizes of (100, 100), but if a dimension is smaller than 100,
    # the chunk size might match the dimension. For example, if x_len=50 < 100, the
    # chunk is effectively (100, 50) or (init_time_len, 50) if init_time_len < 100.
    # So we verify each dimension chunk is the min of chunk request vs. actual size.
    assert chunks[0] == min(100, ds_final.dims["time"])
    assert chunks[1] == min(100, ds_final.dims["x"])

    print(f"Test passed for init_time_len={init_time_len}, update_time_len={update_time_len}")




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
    coords = [
        Coord("time_stamp", max_values=0, description="DateTime stamps of individual time slices. No unit for date_time stamps."),
        Coord("location", max_values=40, struct_cols=("lon", "lat"), description="Location coordinates of measurments."),
        Coord("sensor",  description="Sensor identifier."),
    ]

    ds = create(zarr_path, sample_df, coords)
    # Check that the time coordinate is empty and location coordinate has length 40.
    assert len(ds.dims) == 3
    assert len(ds.coords) == 4
    assert ds.coords["time_stamp"].size == 0
    assert ds.coords["location_x"].size == 40
    assert ds.coords["location_y"].size == 40
    assert ds.coords["sensor"].size == 3


@pytest.mark.skip
def test_pivot_nd():
    # Create a sample Polars DataFrame with columns for a 3D multi-index.
    df = pl.DataFrame({
        "time": [1, 1, 1, 2, 2],
        "loc": ["A", "A", "B", "A", "B"],
        "sensor": ["X", "Y", "X", "X", "Y"],
        "var": [10.0, 11.0, 12.0, 20.0, 21.0]
    })
    dims = ["time", "loc", "sensor"]

    # Call pivot_nd to generate the N-d array and the coordinate mapping.
    arr, unique_coords = pivot_nd(df, dims, "var", fill_value=np.nan)

    # Expected unique coordinates:
    # time: [1, 2]
    # loc:  ["A", "B"]
    # sensor: ["X", "Y"]
    expected_coords = {
        "time": np.array([1, 2]),
        "loc": np.array(["A", "B"]),
        "sensor": np.array(["X", "Y"])
    }

    # Check that unique_coords match.
    for d in dims:
        np.testing.assert_array_equal(unique_coords[d], expected_coords[d])

    # The resulting array should have shape (2, 2, 2)
    assert arr.shape == (2, 2, 2)

    # Expected array construction:
    # For time=1, loc="A", sensor="X": 10.0
    # For time=1, loc="A", sensor="Y": 11.0
    # For time=1, loc="B", sensor="X": 12.0
    # For time=1, loc="B", sensor="Y": missing -> NaN
    # For time=2, loc="A", sensor="X": 20.0
    # For time=2, loc="A", sensor="Y": missing -> NaN
    # For time=2, loc="B", sensor="X": missing -> NaN
    # For time=2, loc="B", sensor="Y": 21.0
    expected_arr = np.array([
        [[10.0, 11.0],
         [12.0, np.nan]],
        [[20.0, np.nan],
         [np.nan, 21.0]]
    ])

    np.testing.assert_allclose(arr, expected_arr, equal_nan=True)




# def update_xarray_dynamic(ds: xr.Dataset, df: pl.DataFrame) -> xr.Dataset:
#     """
#     Update an xarray Dataset in a dynamic way, allowing incomplete updates.
#
#     The dataset has a dynamic (appendable) dimension whose name is stored in ds.attrs["append_dim"]
#     and a fixed dimension "loc". The update is provided as a Polars DataFrame with columns:
#        - the dynamic dimension (e.g. "time")
#        - the fixed dimension "loc"
#        - the data variable "var"
#
#     The DataFrame is pivoted into a table (dynamic x loc) that may have missing values.
#     For each dynamic value:
#        - If the update provides a value, that value is used.
#        - If the update is missing a value (NaN), the current value stored in ds is retained.
#
#     For dynamic values that are not yet present in ds, the dataset is extended (with fill_value NaN)
#     before applying the update.
#     """
#     # Read dynamic dimension name from attributes and set fixed dimension name.
#     append_dim = ds.attrs["append_dim"]
#     fixed_dim = "loc"
#
#     # Pivot the DataFrame.
#     # The resulting table will have dynamic values as index and fixed values as columns.
#     # Missing combinations will be represented as NaN.
#     pivoted = df.pivot(values="var", index=append_dim, columns=fixed_dim)
#
#     # Get the complete set of fixed values from the dataset.
#     ds_locs = ds.coords[fixed_dim].values
#     # Ensure that the pivot table has all fixed-dimension columns; fill missing with NaN.
#     #pdf = pivoted.reindex(columns=ds_locs, fill_value=np.nan)
#
#     # Get the new dynamic values and data.
#     new_times = pivoted.index.to_numpy()  # dynamic dimension update values
#     new_data = pivoted.to_numpy()  # shape (n_new_times, n_locs)
#
#     # Extend the dataset along the dynamic dimension if needed.
#     current_times = ds.coords[append_dim].values
#     mask_missing = ~np.isin(new_times, current_times)
#     missing_times = new_times[mask_missing]
#     if missing_times.size > 0:
#         updated_times = np.concatenate([current_times, missing_times])
#         # Reindex along the dynamic dimension; new rows get fill_value=np.nan.
#         ds = ds.reindex({append_dim: updated_times}, fill_value=np.nan)
#
#     # Now, for the new dynamic times, get the current stored values from ds.
#     # This works both for dynamic values that already existed and those newly appended.
#     current_values = ds["var"].sel({append_dim: new_times, fixed_dim: ds_locs}).values
#
#     # Combine new_data with current_values: where new_data is NaN, keep the current value.
#     combined = np.where(np.isnan(new_data), current_values, new_data)
#
#     # Apply the update in one vectorized assignment.
#     ds["var"].loc[{append_dim: new_times, fixed_dim: ds_locs}] = combined
#
#     return ds

def update_xarray_dynamic(ds: xr.Dataset, df: pl.DataFrame) -> xr.Dataset:
    """
    Update an xarray Dataset in a dynamic way, allowing incomplete updates.

    The dataset uses a dynamic (appendable) dimension whose name is stored in ds.attrs["append_dim"]
    and a fixed dimension "loc". The update is provided as a Polars DataFrame with columns:
       - the dynamic dimension (e.g. "time")
       - the fixed dimension "loc"
       - the data variable "var"

    The DataFrame is pivoted into a (dynamic x loc) table that may contain missing values.
    For each dynamic value, the current stored row is read and combined with the update:
       - Where the update value is not missing, that value is used.
       - Otherwise, the existing value is kept.

    For dynamic values not yet present in ds, the dataset is extended.
    The update is applied in a vectorized manner.
    """
    # Read dimension names.
    append_dim = ds.attrs["append_dim"]  # e.g. "time"
    fixed_dim = "loc"

    # Append missing times in ds

    # Extend the dataset along the dynamic dimension if any new dynamic values are present.
    new_times = df[append_dim].unique().to_numpy()
    current_times = ds.coords[append_dim].values
    mask_missing = ~np.isin(new_times, current_times)
    missing_times = new_times[mask_missing]
    if missing_times.size > 0:
        updated_times = np.concatenate([current_times, missing_times])
        ds = ds.reindex({append_dim: updated_times}, fill_value=np.nan)

    # Pivot the update DataFrame using Polars.
    # We use aggregate_fn="first" assuming unique update values per (append_dim, fixed_dim) pair.
    pivoted = df.pivot(index=append_dim, columns=fixed_dim, values="var")
    # Extract new dynamic values and the 2D update array.
    new_times = pivoted[append_dim].to_numpy()  # shape (n_new_times,)
    new_loc = pivoted[fixed_dim].to_numpy()

    # update ds coords


    # Get the complete set of fixed values from the dataset.
    ds_locs = ds.coords[fixed_dim].values
    # Ensure that the pivoted table contains all ds_locs as columns.
    for loc in ds_locs:
        if loc not in pivoted.columns:
            # Add a column filled with NaN. We use pl.lit(np.nan) to create a column.
            pivoted = pivoted.with_column(pl.lit(np.nan).alias(loc))

    # Reorder the pivoted DataFrame so that the first column is the dynamic dimension,
    # and the remaining columns follow the order in ds.coords["loc"].co

    pivoted = pivoted.select([append_dim] + list(ds_locs))

    # Extract new dynamic values and the 2D update array.
    new_times = pivoted[append_dim].to_numpy()  # shape (n_new_times,)
    # new_data will have shape (n_new_times, n_locs)
    new_data = pivoted.select(list(ds_locs)).to_numpy()


    # Get the current stored values for these dynamic rows.
    current_values = ds["var"].sel({append_dim: new_times, fixed_dim: ds_locs}).values
    # Combine new_data with current_values: where new_data is NaN, keep current_values.
    combined = np.where(np.isnan(new_data), current_values, new_data)

    ds["var"].loc[{append_dim: new_times, fixed_dim: ds_locs}] = combined

    # # Attempt vectorized assignment using xarray's .loc.
    # try:
    # except Exception as e:
    #     # Fallback: update using index positions if .loc assignment fails.
    #     time_vals = ds.coords[append_dim].values
    #     indices = np.searchsorted(time_vals, new_times)
    #     updated_array = ds["var"].data
    #     updated_array[indices, :] = combined
    #     ds["var"].data = updated_array

    return ds

"""
Now make the update function a bit more complex:
- read the name of dependable dimension from "append_dim" attribute
- consider other fixed dimension "loc", the df should contain some 2D subarray (some times) x (some locs), if that 2D array is complete 
"""
# --- Pytest test function for the prototype ---

def test_update_xarray_dynamic():
    # Create an initial dataset.
    # Let's assume the dynamic dimension is "time" (stored in ds.attrs["append_dim"])
    # and the fixed dimension is "loc". For example, ds has times [1, 2] and locs ["A", "B"].
    ds = xr.Dataset(
        data_vars={"var": (("time", "loc"), np.array([[10.0, 11.0, np.nan, np.nan],
                                                      [20.0, 21.0, np.nan, np.nan]]))},
        coords={"time": np.array([1, 2]),
                "loc": np.array(["A", "B", "_1", "_2"])},
        attrs={
            "append_dim": "time",
            "coords_valid_size": dict(time=0, loc=2)
        }
    )

    # Create an update DataFrame.
    # For time=2 (existing), update only "A" (provide 200.0) and leave "B" missing.
    # For time=3 (new), update only "B" (provide 310.0) and leave "A" missing.
    df = pl.DataFrame({
        "time": [2, 3, 3],
        "loc": ["A", "B", "C"],
        "var": [200.0, 310.0, 320.0]
    })

    ds_updated = update_xarray_dynamic(ds, df)

    # The time coordinate should now be [1, 2, 3].
    updated_times = np.sort(ds_updated.coords["time"].values)
    np.testing.assert_array_equal(updated_times, np.array([1, 2, 3]))
    np.testing.assert_array_equal(ds_updated.coords["loc"].values, np.array(['A', 'B', 'C', '_2']))

    # Check the updated rows:
    # - Time 1 remains unchanged: [10.0, 11.0]
    # - Time 2: for "A" updated to 200.0, "B" remains as originally 21.0.
    # - Time 3: for new row, "A" remains NaN (no update provided), "B" is updated to 310.0.
    row1 = ds_updated["var"].sel(time=1).values
    row2 = ds_updated["var"].sel(time=2).values
    row3 = ds_updated["var"].sel(time=3).values

    np.testing.assert_array_equal(row1, np.array([10.0, 11.0, np.nan, np.nan]))
    np.testing.assert_array_equal(row2, np.array([200.0, 21.0, np.nan, np.nan]))
    np.testing.assert_array_equal(row2, np.array([np.nan, 310.0, 320.0, np.nan]))
    # For time 3, check that "A" is NaN and "B" equals 310.0.
    #assert np.isnan(row3[0])
    #assert row3[1] == 310.0



# --- Pytest test function for update ---
@pytest.mark.skip
def test_update(tmp_path, sample_df):
    """
    This test first creates a Zarr store (using create) with multi-index coordinates.
    Then it calls update() twice:
      - The first update appends two rows (for a new time_stamp and another new time_stamp)
        for a single fixed combination.
      - The second update overwrites an existing time_stamp row and appends another new row.
    The test asserts that the dynamic coordinate is extended as needed and that the data
    variables are updated (overwritten when a time_stamp is repeated).
    """
    zarr_path = tmp_path / "test.zarr"
    # Create the initial (empty) store.
    ds = create(zarr_path, sample_df,
                index_cols={"time_stamp": 0, "location": 40, "sensor": None})

    # First update: add two rows for a specific fixed index combination.
    # For simplicity, we use the first (valid) fixed entry for both 'location' and 'sensor'.
    update_df1 = pl.DataFrame({
        "time_stamp": [1, 2],
        "location": [sample_df["location"][0], sample_df["location"][0]],  # use first location for both rows
        "sensor": [sample_df["sensor"][0], sample_df["sensor"][0]],  # use first sensor for both rows
        "var1": [10.0, 20.0],
        "var2": [100, 200]
    })
    ds = update(zarr_path, update_df1)
    # Check that the dynamic coordinate "time_stamp" now has length 2 and values [1, 2]
    np.testing.assert_array_equal(ds.coords["time_stamp"].values, np.array([1, 2]))
    # For the fixed indices, the first valid location is at index 0 and first sensor at index 0.
    assert ds["var1"].values[0, 0, 0] == 10.0
    assert ds["var1"].values[1, 0, 0] == 20.0

    # Second update: overwrite the row with time_stamp=1 and append a new row with time_stamp=3.
    update_df2 = pl.DataFrame({
        "time_stamp": [1, 3],
        "location": [sample_df["location"][0], sample_df["location"][0]],
        "sensor": [sample_df["sensor"][0], sample_df["sensor"][0]],
        "var1": [15.0, 30.0],
        "var2": [150, 300]
    })
    ds = update(zarr_path, update_df2)
    # Now, the dynamic coordinate should have length 3 and be [1,2,3]
    np.testing.assert_array_equal(ds.coords["time_stamp"].values, np.array([1, 2, 3]))
    # The update for time_stamp 1 should be overwritten.
    assert ds["var1"].values[0, 0, 0] == 15.0
    # And the new time_stamp 3 row should be appended.
    assert ds["var1"].values[2, 0, 0] == 30.0


# ---
# Note: The functions _make_coord, _guess_col_dtype and create() are assumed to be defined elsewhere,
# and sample_df is the fixture provided in the question.

@pytest.mark.skip
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
