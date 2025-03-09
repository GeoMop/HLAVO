import pytest
import xarray as xr
import numpy as np
import numpy.testing as npt
import zarr

# Attempt to import dask
try:
    import dask
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def update_zarr_store(zarr_path: str, ds_update: xr.Dataset) -> None:
    """
    Update/append 'ds_update' into an existing Zarr store at 'zarr_path'.
    - Dims: 'time' and 'x'.
    - Possibly extends the 'time' dimension if new time coords go beyond existing range.
    - Uses region='auto' so xarray infers which slices to overwrite/append.
    """
    # --- 1) Identify old vs. new time labels ---
    ds_existing = xr.open_zarr(zarr_path, chunks=None)  # loads metadata
    old_times = ds_existing.time.values  # e.g. shape=(100,)
    new_times = ds_update.time.values  # e.g. shape=(N,)

    # Check which new_times already exist in old_times
    overlap_mask = np.isin(new_times, old_times)
    # Overlap portion
    ds_overlap = ds_update.sel(time=new_times[overlap_mask])
    # Pure extension portion
    ds_extend = ds_update.sel(time=new_times[~overlap_mask])

    # --- 2) Overlap: region="auto" update ---
    # region="auto" works ONLY if all time coords in `ds_overlap` exist in the store
    # (which they do, by construction).
    if ds_overlap.sizes.get("time", 0) > 0:
        ds_overlap.to_zarr(
            zarr_path,
            mode="r+",  # read/write
            region="auto"  # Overwrite the existing time indices in the store
        )

    # --- 3) Extension: append_dim="time" ---
    # appending is valid only for new time coords that aren't in the store
    # (which they aren't, by construction).
    if ds_extend.sizes.get("time", 0) > 0:
        ds_extend.to_zarr(
            zarr_path,
            mode="a",  # append mode
            append_dim="time"  # dimension to grow
        )



def update_zarr_loop(zarr_path: str, ds_update: xr.Dataset, dims_order=None) -> dict:
    """
    Iteratively update/append ds_update into an existing Zarr store at zarr_path.

    This function works in two phases:

    Phase 1 (Dive):
      For each dimension in dims_order (in order):
        - Split ds_update along that dimension into:
            • overlap: coordinate values that already exist in the store.
            • extension: new coordinate values.
        - Save the extension subset (per dimension) for later appending.
        - For subsequent dimensions, keep only the overlapping portion.

    Phase 2 (Upward):
      - Write the final overlapping subset using region="auto".
      - Then, in reverse order, for each dimension that had an extension:
            • Reindex the corresponding extension subset so that for all dimensions
              except the current one the coordinate values come from the store.
            • Append that reindexed subset along the current dimension.
            • Update the merged coordinate for that dimension.

    Parameters
    ----------
    zarr_path : str
        Path to the existing Zarr store.
    ds_update : xr.Dataset
        The dataset to update. Its coordinate values in one or more dimensions may be new.
    dims_order : list of str, optional
        The list of dimensions to process (in order). If None, defaults to list(ds_update.dims).

    Returns
    -------
    merged_coords : dict
        A dictionary mapping each dimension name to the merged (updated) coordinate array.
    """
    if dims_order is None:
        dims_order = list(ds_update.dims)

    # Open store (we assume the dimension already exists)
    ds_existing = xr.open_zarr(zarr_path, chunks=None)
    # --- Phase 1: Dive (split by dimension) ---
    # We create a dict to hold the extension subset for each dimension.
    ds_extend_dict = {}
    # And we will update ds_current to be the overlapping portion along all dims processed so far.
    ds_overlap = ds_update.copy()
    for dim in dims_order:
        if dim not in ds_existing.dims:
            raise ValueError(f"Dimension '{dim}' not found in the existing store.")
        old_coords = ds_existing[dim].values
        new_coords = ds_overlap[dim].values

        # Determine which coordinates in ds_current already exist.
        overlap_mask = np.isin(new_coords, old_coords)
        ds_extend_dict[dim] = ds_overlap.sel({dim: new_coords[~overlap_mask]})
        ds_overlap = ds_overlap.sel({dim: new_coords[overlap_mask]})

    # At this point, ds_overlap covers only the coordinates that already exist in the store
    # in every dimension in dims_order. Write these (overlapping) data using region="auto".
    update_overlap_size = np.prod(list(ds_overlap.sizes.values()))
    if update_overlap_size > 0:
        ds_overlap.to_zarr(zarr_path, mode="r+", region="auto")

    # --- Phase 2: Upward (process extension subsets in reverse order) ---
    # We also update a merged_coords dict from the store.
    merged_coords = {d: ds_existing[d].values for d in ds_existing.dims}

    # Loop upward in reverse order over dims_order.
    for dim in reversed(dims_order):
        ds_ext = ds_extend_dict[dim]
        if ds_ext is None or ds_ext.sizes.get(dim, 0) == 0:
            continue  # No new coordinates along this dimension.

        # For all dimensions other than dim, reindex ds_ext so that the coordinate arrays
        # come from the store (i.e. the full arrays). This ensures consistency.
        # (This constructs an indexers dict using the existing merged coordinates.)
        indexers = {d: merged_coords[d] for d in ds_ext.dims if d != dim}
        ds_ext_reindexed = ds_ext.reindex(indexers, fill_value=np.nan)

        # Append the extension subset along the current dimension.
        ds_ext_reindexed.to_zarr(zarr_path, mode="a", append_dim=dim)

        # Update merged coordinate for dim: concatenate the old coords with the new ones.
        new_coords_for_dim = ds_ext[dim].values
        merged_coords[dim] = np.concatenate([merged_coords[dim], new_coords_for_dim])

    return merged_coords


# =============================================================================
# Example test using the two-phase update.
# =============================================================================
def test_extension_multidim(tmp_path):
    """
    Test scenario:
      - Create an initial store with dimensions:
          time: 0..99  and  x: 0..49.
      - ds_update covers:
          time: 90..119   and   x: 40..59.
        (For 'time': overlap = 90..99, extension = 100..119.
         For 'x': overlap = 40..49, extension = 50..59.)
      - Final expected store shape:
          time: 120  and  x: 60.
    """
    # --- Step 1. Create initial dataset ---
    init_time_len = 100
    x_len = 50
    t_init = np.arange(init_time_len)
    x_init = np.arange(x_len)
    data_init = np.random.rand(init_time_len, x_len)
    ds_init = xr.Dataset(
        {"var": (("time", "x"), data_init)},
        coords={"time": t_init, "x": x_init}
    )
    zpath = str(tmp_path / "test.zarr")
    ds_init.to_zarr(zpath, mode="w", compute=False)

    # --- Step 2. Create update dataset ---
    # We want to update with:
    # time: 90..119  (so overlap = 90..99, extend = 100..119)
    # x: 40..59      (so overlap = 40..49, extend = 50..59)
    t_new = np.arange(90, 120)
    x_new = np.arange(40, 60)
    data_update = np.random.rand(len(t_new), len(x_new))
    ds_update = xr.Dataset(
        {"var": (("time", "x"), data_update)},
        coords={"time": t_new, "x": x_new}
    )

    # --- Step 3. Call the loop-based updater ---
    merged_coords = update_zarr_loop_two_phase(zpath, ds_update, dims_order=["time", "x"])

    # --- Step 4. Verify final store ---
    ds_final = xr.open_zarr(zpath)
    assert ds_final.dims["time"] == 120  # Expect times 0..119
    assert ds_final.dims["x"] == 60  # Expect x 0..59

    print("Test passed: final shape is time=120, x=60")


def update_zarr_recursion(
    zarr_path: str,
    ds_update: xr.Dataset,
    dims_order=None
):
    """
    Hybrid dimension-by-dimension update:
      - For each dimension in dims_order:
        1) We split ds_update into Overlap (coords existing in store) vs. Extension (coords new to store).
        2) We recurse ONLY on Overlap, so deeper dimensions can also split if needed.
        3) We do an immediate append+final-write for Extension, so we do NOT recurse on newly appended data.

    If there are no more dimensions to process (base case),
    we do a single region='auto' write of ds_update.

    Parameters
    ----------
    zarr_path : str
        Path to the existing Zarr store.
    ds_update : xr.Dataset
        The dataset containing possibly overlapping + new coords.
    dims_order : list of str, optional
        The dimensions to process in order. If None, defaults to ds_update.dims in the order they appear.
    """

    # Base case: if no more dims to process, just do one final region='auto' write
    if dims_order is None:
        dims_order = list(ds_update.dims)
    if not dims_order:
        # Write part that overlaps in all dimensions
        ds_update.to_zarr(zarr_path, mode="r+", region="auto")
        return {d: ds_update[d].values for d in ds_update.dims}

    # Take the first dimension to process
    dim = dims_order[0]

    # Open store to see existing coords for this dimension
    ds_existing = xr.open_zarr(zarr_path, chunks=None)
    if dim not in ds_existing.dims:
        raise ValueError(f"Dimension '{dim}' not found in existing store.")

    old_coords = ds_existing[dim].values
    new_coords = ds_update[dim].values

    # Split ds_update into overlap vs extension for this dimension
    overlap_mask = np.isin(new_coords, old_coords)
    ds_overlap = ds_update.sel({dim: new_coords[overlap_mask]})
    ds_extend = ds_update.sel({dim: new_coords[~overlap_mask]})
    merged_coords = np.concatenate([old_coords, new_coords[~overlap_mask]])

    # ------------------
    # 1) Overlap portion: Recurse
    # ------------------
    if ds_overlap.sizes.get(dim, 0) > 0:
        existing_dims = update_zarr_recursion(
            zarr_path,
            ds_overlap,
            dims_order=dims_order[1:]
        )
    else:
        existing_dims = {d: ds_update[d].values for d in ds_update.dims}
    # ------------------
    # 2) Extension portion: Append + final write, no recursion
    # ------------------
    if ds_extend.sizes.get(dim, 0) > 0:
        # Append new coords along `dim`
        indexers = {d: existing_dims[d] for d in ds_extend.dims if d != dim}
        ds_reindexed = ds_extend.reindex(indexers=indexers)
        ds_reindexed.to_zarr(zarr_path, mode="a", append_dim=dim)

        # Now do a final region='auto' write for these newly appended coords
        #ds_store_after_append = xr.open_zarr(zarr_path, chunks=None)
        #ds_new_slice = ds_store_after_append.sel({dim: ds_extend[dim]})
        #ds_new_slice.to_zarr(zarr_path, mode="r+", region="auto")
    return {d: merged_coords if d == dim else existing_dims[d] for d in ds_update.dims}



def zarr_kwargs():
    # Only set chunk encoding if Dask is installed
    if HAS_DASK:
        encoding = {"var": {"chunks": (100, 100)}}
        kwargs = {"encoding": encoding, 'compute': False, 'write_empty_chunks': False}
    else:
        kwargs = {}
    return kwargs

def init_ds(tmp_path, time_len, x_len):
    # Create initial data so we can test final results
    data_initial = np.random.rand(time_len, x_len)
    ds_initial = xr.Dataset(
        {
            "var": (("time", "x"), data_initial)
        },
        coords={
            "time": np.arange(time_len),  # 0..(init_time_len - 1)
            "x": np.arange(x_len)
        }
    )

    zarr_path = str(tmp_path / "test.zarr")

    # -------------------------
    # 1. WRITE METADATA ONLY
    # -------------------------
    ds_initial.to_zarr(
        zarr_path,
        mode="w",        # create or overwrite store
        **zarr_kwargs()
    )
    return zarr_path, data_initial


@pytest.mark.parametrize("init_time_len, update_time_len", [(100, 50)])
def test_xarray_zarr_append_and_update(tmp_path, init_time_len, update_time_len, x_len=50):
    """
    1. Create initial dataset with shape (time=init_time_len, x=50).
       If Dask is available, we chunk at (100,100).
    2. Write Zarr *metadata only* (compute=False).
    3. Use 'update_zarr_store' to append a second dataset with shape
       (time=update_time_len, x=50).
    4. Verify final shape.
    5. Verify final data values.
    6. If Dask is available, assert the underlying Zarr chunk sizes.
    """

    zarr_path, data_initial = init_ds(tmp_path, init_time_len, x_len)
    # ----------------------------
    # 2. UPDATE / APPEND NEW DATA
    # ----------------------------
    new_time_start = init_time_len
    new_time_stop = init_time_len + update_time_len
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

    #update_zarr_store(zarr_path, ds_update)
    update_zarr_loop(zarr_path, ds_update, dims_order=["time", "x"])
    # -----------------------
    # 3. VERIFY FINAL ZARR
    # -----------------------
    ds_final = xr.open_zarr(zarr_path)

    expected_time_len = init_time_len + update_time_len
    assert ds_final.dims["time"] == expected_time_len
    assert ds_final.dims["x"] == x_len

    # -----------------------
    # 4. VERIFY DATA VALUES
    # -----------------------
    final_initial_part = ds_final["var"].isel(time=slice(0, init_time_len)).values
    npt.assert_allclose(final_initial_part, data_initial)

    final_appended_part = ds_final["var"].isel(time=slice(init_time_len, expected_time_len)).values
    npt.assert_allclose(final_appended_part, data_update)

    # -----------------------
    # 5. CHECK CHUNK SIZES
    # -----------------------
    # Only check chunk sizes if Dask is installed and we used chunking.
    if HAS_DASK:
        zgroup = zarr.open_group(zarr_path, mode="r")
        zvar = zgroup["var"]
        chunks = zvar.chunks

        # We asked for (100, 100). If the dimension is smaller than 100,
        # the chunk size matches the dimension's size.
        # For time and x, we check min(100, actual_dim_len).
        assert chunks[0] == min(100, expected_time_len)
        assert chunks[1] == min(100, x_len)
        print("Chunk-size check passed (Dask available).")

    print("Test passed for init_time_len={}, update_time_len={}".format(
        init_time_len, update_time_len
    ))


def test_extension_multidim(tmp_path):
    """
    Example test: partial overlap in 'time' + 'x' plus new coords in each dimension.
    Overlap portion recurses on next dims, extension portion appends once and does a final write.
    """
    init_time_len = 100
    x_len = 50
    zpath, data_initial = init_ds(tmp_path, init_time_len, x_len)

    # 2) ds_update with partial overlap + extension in time and x
    #    e.g., time=90..119, x=40..59
    t_new = np.arange(90, 120)  # Overlap=90..99, Extend=100..119
    x_new = np.arange(40, 60)   # Overlap=40..49, Extend=50..59
    data_update = np.random.rand(len(t_new), len(x_new))
    ds_update = xr.Dataset(
        {
            "var": (("time", "x"), data_update)
        },
        coords={
            "time": t_new,
            "x": x_new,
        }
    )

    # 3) Call our hybrid update
    update_zarr_loop(zpath, ds_update, dims_order=["time","x"])

    # 4) Verify final store
    ds_final = xr.open_zarr(zpath)
    assert ds_final.dims["time"] == 120  # 0..119
    assert ds_final.dims["x"] == 60      # 0..59

    print("Test passed: partial overlap recursed, extension done once per dimension.")