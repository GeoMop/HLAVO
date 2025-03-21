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




def zarr_kwargs():
    # Only set chunk encoding if Dask is installed
    if HAS_DASK:
        encoding = {"var": {"chunks": (128, 128)}}
        kwargs = {"encoding": encoding, 'compute': False}
    else:
        kwargs = {}
    return kwargs


def ds_block(time_range, x_range):
    t_coords = np.arange(*time_range)  # Overlap=90..99, Extend=100..119
    x_coords = np.arange(*x_range)   # Overlap=40..49, Extend=50..59
    data = np.random.rand(len(t_coords), len(x_coords))
    ds = xr.Dataset(
        {
            "var": (("time", "x"), data)
        },
        coords={
            "time": t_coords,
            "x": x_coords
        }
    )
    return ds

def init_ds(tmp_path, time_len, x_len):
    # Create initial data so we can test final results
    ds_initial = ds_block((0, time_len), (0, x_len))
    zarr_path = str(tmp_path / "test.zarr")

    # -------------------------
    # 1. WRITE METADATA ONLY
    # -------------------------
    ds_initial.to_zarr(
        zarr_path,
        mode="w",        # create or overwrite store
        **zarr_kwargs()
    )
    return zarr_path, ds_initial['var'].to_numpy()

@pytest.mark.parametrize("init_time_len, update_time_len", [(1000, 500)])
def test_xarray_zarr_append_and_update(tmp_path, init_time_len, update_time_len, x_len=5000):
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
    overlap = 10
    new_time_start = init_time_len - overlap
    new_time_stop = init_time_len + update_time_len
    ds_update = ds_block((new_time_start, new_time_stop), (0, x_len))
    data_update = ds_update['var'].to_numpy()

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
    final_initial_part = ds_final["var"].isel(time=slice(0, init_time_len-overlap)).values
    npt.assert_allclose(final_initial_part, data_initial[:-overlap], rtol=1e-5, atol=1e-10)

    final_appended_part = ds_final["var"].isel(time=slice(init_time_len-overlap, expected_time_len)).values
    npt.assert_allclose(final_appended_part, data_update, rtol=1e-5, atol=1e-10)

    # -----------------------
    # 5. CHECK CHUNK SIZES
    # -----------------------
    # Only check chunk sizes if Dask is installed and we used chunking.
    if HAS_DASK:
        zgroup = zarr.open_group(zarr_path, mode="r")
        zvar = zgroup["var"]
        chunk_size_t, chunk_size_x = zarr_kwargs()["encoding"]["var"]["chunks"]
        assert (chunk_size_t, chunk_size_x) == zvar.chunks
        # We asked for (100, 100). If the dimension is smaller than 100,
        # the chunk size matches the dimension's size.
        # For time and x, we check min(100, actual_dim_len).
        n_chunks = zvar.nchunks
        assert n_chunks[0] == expected_time_len / chunk_size_t
        assert n_chunks[1] == x_len / chunk_size_x
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
    time_max = 1000
    x_max = 200
    ds_update_1 = ds_block((90, time_max), (40, x_max))

    # 3) Call our hybrid update
    update_zarr_loop(zpath, ds_update_1, dims_order=["time","x"])

    # 4) Verify final store
    ds_final = xr.open_zarr(zpath)
    assert ds_final.dims["time"] == time_max  # 0..119
    assert ds_final.dims["x"] == x_max      # 0..59

    time_max = 1000
    x_max = 1000
    ds_update_2 = ds_block((90, time_max), (150, x_max))

    # 3) Call our hybrid update
    update_zarr_loop(zpath, ds_update_2, dims_order=["time","x"])

    print("Test passed: partial overlap recursed, extension done once per dimension.")


