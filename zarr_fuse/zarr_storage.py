from typing import List, Callable, Dict, Optional
from pathlib import Path
import polars as pl
import zarr
import xarray as xr
import json
import numpy as np
import attrs
from pint import  Unit
import zarr_structure as struc
from anytree import NodeMixin


"""

"""

from anytree import NodeMixin
import xarray as xr
import numpy as np
import zarr


class Node:
    """
    A lazy node representing a dataset stored in a single Zarr storage.
    We flatten the tree structure representing every node by a key equal to the node's path.
    (Note: if the length of the ptah would be the problem, we can have a dict attribute asigning path -> hash)
    Each path starts with and is separated by "/", root node name is "/".

    Each node’s dataset is stored under a group defined by a "/"‑separated relative path.

    Specific DataArray as Coord attributes:
    - chunk_size: int
    - description: str
    - unit: str (interpretable by Pint)
    - df_cols: List[str] (column names in the DataFrame)

    Specific DataArray as Quantity attributes:
    - description: str
    - unit: str (interpretable by Pint)
    - df_cols: List[str] (column names in the DataFrame)
    """
    PATH_SEP = "/"

    @classmethod
    def create_storage(cls, structure:struc.ZarrNodeStruc, zarr_store) -> 'Node':
        """
        Consturct an empty zarr tree storage in the 'zarr_store' with
        tree structure and node DataSets given by the 'description',
        zarr_store is an instance of LocalStore, RemoteStore, MemoryStore, ...
        but could be also path on local or remote file system.
        TODO: test and better document
        Returns:
        root Node
        """
        root = cls("", zarr_store)
        zarr.group(store=zarr_store, overwrite=True)
        root.initialize_node(structure)
        return root

    def initialize_node(self, structure:struc.ZarrNodeStruc):
        """ Write node to ZARR sotrage and create childs."""
        empty_ds = Node.empty_ds(structure)
        self.write_ds(empty_ds)
        for key in structure:
            if key in struc.reserved_keys:
                continue
            child = self._add_node(key)
            child.initialize_node(structure[key])

    @staticmethod
    def _create_coords(coords: List[struc.Coord]) -> xr.Coordinates:
        """
        Create xarray Coordinates from a dictionary of Coord objects using
        the explicit Coordinates constructor. Each coordinate is stored as an xarray.Variable.

        If a Coord has no `values`, an empty NumPy array is used.
        The chunk_size attribute is also stored.

        Parameters:
            coord_dict (Dict[str, Coord]): Mapping of coordinate names to Coord objects.

        Returns:
            Coordinates: An xarray Coordinates object built from the provided variables.
        """

        coord_vars = {
            coord.name:
            xr.Variable(
                dims=(coord.name,),  # Default dimension named after the coordinate key.
                data=coord.values,
                attrs={
                    "unit": coord.unit,
                    "description": coord.description,
                    "df_cols": coord.df_cols,
                    "chunk_size": coord.chunk_size,
                }
            )
            for coord in coords
        }
        return xr.Coordinates(coord_vars)

    @staticmethod
    def empty_ds(description):
        """
        Create an *empty* Zarr storage for multiple indices given by index_col.
        - first index is dynamic, e.g. time, new times are appended by update function
        - other indices are fixed, e.g. location, new locations are not allowed, all locations must be provided
          by the passed DF. However
           !! Need some mean to trace unused indices, relying on placehloders not possible for general types.

        - 'df' is used ONLY for column names/types (besides 'time_stamp' and 'location').
        - index_cols, [time_col, ...]
        - 'max_locations' is the fixed size for the location dimension.
        - idx_ranges reserved limits for each index column, the first index is just initali limit, could be appended.
        - Store an empty location_map in dataset attributes.
        - **kwargs can include chunking or compression settings passed to to_zarr().
          ... chunking coud either go to ds.chunks(...) or thrugh kwargs['enconding']['time'] = {'chunksizes': (1000, 10)}

        df, index_cols, iterators
        """
        # Identify the data columns (excluding time_stamp, location)

        # Build an Xarray Dataset with dims: (time=0, location=max_locations)
        # We'll create a coordinate for "time" (initially empty),
        # and a coordinate for "location" (just an integer range 0..max_locations-1).
        coords_obj = Node._create_coords(description['COORDS'])

        shape = lambda var: [coords_obj[c].shape[0] for c in var.coords]
        data_vars = {
            var.name:
                xr.Variable(
                    dims=tuple(var.coords),  # Default dimension named after the coordinate key.
                    data=np.empty(shape(var), dtype=float),
                    attrs={
                        "unit": var.unit,
                        "description": var.description,
                        "df_cols": var.df_cols
                    }
                )
            for var in description['VARS']
        }
        attrs = description.get('ATTRS', {})
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords_obj,
            attrs=attrs
        )
        return ds

    @classmethod
    def read_store(cls, zarr_store):
        """
        Reconstruct the tree from a single Zarr store.

        Parameters:
          storage_url (str): The URL or path to the Zarr storage.

        Returns:
          Node: The root node of the reconstructed tree.

        Assumes that the root node is stored at the root group (i.e. with group path "").
        """
        root = cls("", store=zarr_store)
        root._load_children()
        return root

    def _add_node(self, name):
        assert name not in self.children
        node = Node(name, self.store, self)
        self.children[name] = node
        return node

    def _storage_group_paths(self):
        path = self.group_path.strip(self.PATH_SEP)
        root_group = zarr.open_group(self.store, path=path, mode='r')
        sub_groups = list(root_group.groups())
        return sub_groups

    def _load_children(self):
        """
        Recursively find and attach child nodes from the store for this node.
        A child is detected if there exists a key of the form "{self.group_path}/{child}/.zgroup".
        """
        for key, group in self._storage_group_paths():
            child = self._add_node(key)
            child._load_children()


    def __init__(self, name, store, parent=None):
        """
        Parameters:
          name (str): The name of the node. For the root node, use an empty string ("").
          store (MutableMapping): The underlying Zarr store.
          parent (Node, optional): Parent node.
        """
        self.name = name
        self.store = store
        self.parent = parent
        self.children = {}


    @property
    def _path_list(self):
        if self.parent is None:
            return []
        else:
            return [*self.parent._path_list, self.name]

    @property
    def group_path(self):
        """
        Compute the group path for this node from its parent's group_path and its name.
        For the root node (parent is None), returns "" (empty string).
        For child nodes, returns a string like "child" or "parent/child".
        """
        return Node.PATH_SEP.join(self._path_list)

    def items(self):
        return self.children.items()
        # for key, item in self.childs:
        #     gp = self.group_path
        #     if key.startswith(self.group_path):
        #         key = key[len(gp):]
        #         end_child_name = key.find(Node.PATH_SEP)
        #         end_child_name = len(key) if end_child_name == -1 else end_child_name
        #         key = key[:end_child_name]
        #         yield key, item


    @property
    def dataset(self):
        """
        Lazily open the dataset stored in this node's group.
        Returns a (possibly dask‑backed) xarray.Dataset.
        """
        rel_path = self.group_path #+ self.PATH_SEP + "dataset"
        rel_path = rel_path.strip(self.PATH_SEP)
        return xr.open_zarr(self.store, group=rel_path)

    def update(self, polars_df):
        """
        Atomically update this node's dataset using a Polars DataFrame.

        Parameters:
          polars_df: A Polars DataFrame (or substitute) with new data.
          dataset_update_func: A function taking (dataset, polars_df) and returning an updated dataset.

        The updated dataset is written (overwriting) to the Zarr group.
        TODO:
        - loop update approach modified using:
           - pivot_nd for DF -> DS conversion, should be a method that uses coords and df_cols
           -
        """
        ds = self.pivot_nd(polars_df)
        return self.update_zarr_loop(ds)

    def pivot_nd(self, df: pl.DataFrame, fill_value=np.nan):
        """
        Pivot a Polars DataFrame with columns for each dimension in self.dataset.dims
        and one value column (per variable) into an N-dim xarray.Dataset.

        For each dimension, the coordinate values are taken as the intersection between
        the dataset’s coordinate (self.dataset[d].values) and the unique values in df
        (from the column specified by self.dataset[d].attrs['df_cols'][0]). The intersection
        is done in a vectorized way and preserves the order given by the dataset coordinate.

        For each variable in self.dataset.variables, values from the corresponding df column
        are inserted into an output array of shape determined by the sizes of the common
        coordinate sets. If duplicate keys occur, later values win.

        Returns:
            xr.Dataset: The pivoted dataset with data variables defined on the new N-dim grid,
                        and with coordinates for each dimension.
        """
        idx_list = []
        coord_sizes = []
        coords_dict = {}
        dims = tuple(self.dataset.dims.keys())
        # Loop over each dimension in the original dataset.
        for d in dims:
            # Get the name(s) of the column(s) in df corresponding to this dimension.
            df_cols = self.dataset[d].attrs['df_cols']
            # Get the coordinate values from the DataFrame column.
            # TODO: deal with df_cols as list of columns, hash tuples
            df_coord_array = df[df_cols[0]].to_numpy()
            # Get the coordinate values from the dataset (assumed to be in desired order).
            coords = np.sort(np.unique(df_coord_array))
            coords_dict[d] = coords  # will be used as the coordinate values for this dim.
            coord_sizes.append(len(coords))

            # Map each row’s coordinate (from df) to its index in the common_coords.
            # (This works as long as common_coords is sorted. In many cases ds coordinates are already sorted.)
            final_idx = np.searchsorted(coords, df_coord_array)
            idx_list.append(final_idx)

        # Create an array of indices for all dimensions, one row per dimension.
        idx_arr = np.vstack(idx_list)
        # Compute flat indices for the N-dim output array.
        flat_idx = np.ravel_multi_index(idx_arr, dims=coord_sizes)

        # Helper: for a given variable, build the pivoted array.
        def values(var_name):
            col_name = self.dataset[var_name].attrs['df_cols'][0]
            column_vals = df[col_name].to_numpy()
            # Choose a dtype based on the column (floating or object)
            dtype = column_vals.dtype if np.issubdtype(column_vals.dtype, np.floating) else object
            # Create an empty output array with fill_value.
            result = np.full(coord_sizes, fill_value, dtype=dtype)
            # Fill in the values; if duplicate keys occur, later values win.
            result.flat[flat_idx] = column_vals
            return (dims, result)

        # Create DataArrays for each variable using the dims specified in self.dataset.
        data_vars = {var: values(var) for var in self.dataset.data_vars}

        # Build and return the xarray.Dataset with the computed coordinates.
        ds_out = xr.Dataset(data_vars=data_vars, coords=coords_dict)
        return ds_out

    def write_ds(self, ds, **kwargs):
        rel_path = self.group_path # + self.PATH_SEP + "dataset"
        rel_path = rel_path.strip(self.PATH_SEP)
        #path_store = zarr.open_group(self.store, mode=mode, path=rel_path)
        #ds.to_zarr(path_store,  **kwargs)
        ds.to_zarr(self.store, group = rel_path, mode="a", consolidated=False, **kwargs)

    def update_zarr_loop(self, ds_update: xr.Dataset) -> xr.Dataset:
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
        ds_existing = self.dataset
        # --- Phase 1: Dive (split by dimension) ---
        # We create a dict to hold the extension subset for each dimension.
        ds_extend_dict = {}
        ds_overlap = ds_update.copy()
        dims_order = tuple(ds_update.dims.keys())
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
            self.write_ds(ds_overlap, region="auto")

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
            self.write_ds(ds_ext_reindexed, append_dim=dim)

            # Update merged coordinate for dim: concatenate the old coords with the new ones.
            new_coords_for_dim = ds_ext[dim].values
            merged_coords[dim] = np.concatenate([merged_coords[dim], new_coords_for_dim])

        return merged_coords

    # def update_zarr_loop(self, ds_existing: xr.Dataset, df_update: pl.DataFrame) -> dict:
    #     """
    #     Iteratively update/append ds_update into an existing Zarr store at zarr_path.
    #
    #     This function works in two phases:
    #
    #     Phase 1 (Dive):
    #       For each dimension in dims_order (in order):
    #         - Split ds_update along that dimension into:
    #             • overlap: coordinate values that already exist in the store.
    #             • extension: new coordinate values.
    #         - Save the extension subset (per dimension) for later appending.
    #         - For subsequent dimensions, keep only the overlapping portion.
    #
    #     Phase 2 (Upward):
    #       - Write the final overlapping subset using region="auto".
    #       - Then, in reverse order, for each dimension that had an extension:
    #             • Reindex the corresponding extension subset so that for all dimensions
    #               except the current one the coordinate values come from the store.
    #             • Append that reindexed subset along the current dimension.
    #             • Update the merged coordinate for that dimension.
    #
    #     Parameters
    #     ----------
    #     zarr_path : str
    #         Path to the existing Zarr store.
    #     ds_update : xr.Dataset
    #         The dataset to update. Its coordinate values in one or more dimensions may be new.
    #     dims_order : list of str, optional
    #         The list of dimensions to process (in order). If None, defaults to list(ds_update.dims).
    #
    #     Returns
    #     -------
    #     merged_coords : dict
    #         A dictionary mapping each dimension name to the merged (updated) coordinate array.
    #     """
    #
    #     # --- Phase 1: Dive (split by dimension) ---
    #     # We create a dict to hold the extension subset for each dimension.
    #     df_extend = {}
    #     # And we will update ds_current to be the overlapping portion along all dims processed so far.
    #     df_overlap = df_update.copy()
    #     for dim in dims_order:
    #         if dim not in ds_existing.dims:
    #             raise ValueError(f"Dimension '{dim}' not found in the existing store.")
    #         old_coords = ds_existing[dim].values
    #         new_coords = df_overlap[dim].values
    #
    #         # Determine which coordinates in ds_current already exist.
    #         overlap_mask = np.isin(new_coords, old_coords)
    #         df_extend[dim] = df_overlap[~overlap_mask]
    #         df_overlap = df_overlap[overlap_mask]
    #
    #     overlap_size = np.prod([ len(df_overlap[dim].unique()) for dim in dims_order])
    #
    #     # At this point, ds_overlap covers only the coordinates that already exist in the store
    #     # in every dimension in dims_order. Write these (overlapping) data using region="auto".
    #     if overlap_size > 0:
    #         ds_overlap = pivot_nd(df_overlap, dims_order)
    #         ds_overlap.to_zarr(self.store, mode="r+", region="auto")
    #
    #     # --- Phase 2: Upward (process extension subsets in reverse order) ---
    #     # Initialize a merged_coords dict for actual cords of ds_overlap.
    #     merged_coords = {d: ds_existing[d].values for d in ds_existing.dims}
    #
    #     # Loop upward in reverse order over dims_order.
    #     for dim in reversed(dims_order):
    #         df_ext = df_extend[dim]
    #         if df_ext is None or df_ext.sizes.get(dim, 0) == 0:
    #             continue  # No new coordinates along this dimension.
    #
    #         # For all dimensions other than dim, reindex ds_ext so that the coordinate arrays
    #         # come from the store (i.e. the full arrays). This ensures consistency.
    #         # (This constructs an indexers dict using the existing merged coordinates.)
    #         indexers = {d: merged_coords[d] for d in df_ext.dims if d != dim}
    #         ds_ext_reindexed = df_ext.reindex(indexers, fill_value=np.nan)
    #
    #         # Append the extension subset along the current dimension.
    #         ds_ext_reindexed.to_zarr(zarr_path, mode="a", append_dim=dim)
    #
    #         # Update merged coordinate for dim: concatenate the old coords with the new ones.
    #         new_coords_for_dim = ds_ext[dim].values
    #         merged_coords[dim] = np.concatenate([merged_coords[dim], new_coords_for_dim])
    #
    #     return merged_coords

    # def add_child(self, child_name, dataset=None, **to_zarr_kwargs) -> 'Node':
    #     """
    #     Add a child node to the "in memory" tree representation.
    #     Keep sync with the storage:
    #     - if group exists already and dataset is None, create the node
    #       for the existing group; if dataset is not None, raise excaption
    #     - if group does not exist, create it with given dataset (possibly empty)
    #     The dataset is immediately written to the Zarr store under a group defined by this node and the child's name.
    #
    #     Parameters:
    #       child_name (str): The child's name.
    #       dataset (xr.Dataset): The dataset to store.
    #       **to_zarr_kwargs: Additional keyword arguments to pass to to_zarr.
    #
    #     Returns:
    #       Node: The new child node.
    #     """
    #     if child_name in self.childs:
    #         raise KeyError("Child node already exists.")
    #     if dataset is None:
    #         # Read from storage
    #         assert self.store.keys
    #     else:
    #     node = Node(child_name, self.store, parent=self)
    #     dataset.to_zarr(self.store, group=node.group_path, mode="w", **to_zarr_kwargs)
    #     return node

    # def delete_node(self):
    #     """
    #     Atomically delete this node from the Zarr storage.
    #     This removes all keys in the store that begin with this node's group path.
    #     Then it removes this node from its parent's children.
    #     """
    #     prefix = self.group_path
    #     prefix = prefix + "/" if prefix else ""
    #     keys_to_delete = [key for key in list(self.store.keys()) if key.startswith(prefix)]
    #     if self.group_path in self.store:
    #         keys_to_delete.append(self.group_path)
    #     for key in keys_to_delete:
    #         del self.store[key]
    #     if self.parent is not None:
    #         self.parent.children = tuple(child for child in self.parent.children if child is not self)
    #






#
#
# def _guess_col_dtype(pl_dtype):
#     """
#     Dispatch by polars value types to treat coords properly.
#     Returns corresponding numpy array type, initial value to fill as a reseved coordinate value, incremental funcion to get
#     further reserved values.
#     Args:
#         pl_dtype:
#
#     Returns:
#         (numpy_type, zero_value, increment_fn)
#
#     """
#     # Very simplistic mapping
#     if isinstance(pl_dtype, pl.Struct):
#         subtypes = [(field.name, *_guess_col_dtype(field.dtype)) for field in pl_dtype.fields]
#         names, types, zeros, inc_fns = zip(*subtypes)
#         np_dtype = np.dtype(list(zip(names, types)))
#
#         def _struct_inc_fn(struct_x):
#             assert len(struct_x) == len(names)
#             inc_values = (inc(val) for inc, val in zip(inc_fns, struct_x.values()))
#             return dict(zip(names, inc_values))
#
#         return (np_dtype,
#                 dict(zip(names, zeros)),
#                 _struct_inc_fn)
#     #if isinstance(pl_dtype, pl.List):
#     #    sub_dtype, zero, inc_fn = _guess_col_dtype(pl_dtype.element_type)
#     #    return np.dtype(sub_dtype), zero, lambda x: [inc_fn(val) for val in x]
#     if pl_dtype in [pl.Float32, pl.Float64]:
#         return np.float64, 0.0, lambda x: x + 1.0
#     elif pl_dtype in [pl.Boolean, pl.Int32, pl.Int64]:
#         return np.int64, -1, lambda x: x - 1
#     elif pl_dtype == pl.Utf8:
#         return np.str_, "_123456", lambda x: f"_{str(hash(hash(x[1:]) + 1))[:6]}"  # or a fixed-length string if you prefer
#     elif (pl_dtype == np.datetime64) or (pl_dtype == pl.Datetime):
#         return np.datetime64, np.datetime64("2021-01-01"), lambda x: x + np.timedelta64(1, "D")
#     #elif pl_dtype == pl.Date or pl_dtype == pl.Datetime:
#     #    return "datetime64[ns]"
#     raise ValueError(f"Unsupported Polars dtype: {pl_dtype}")
#


# def create(zarr_path: Path, df: pl.DataFrame, coords: List[struc.Coord], **kwargs):
#     """
#     Create an *empty* Zarr storage for multiple indices given by index_col.
#     - first index is dynamic, e.g. time, new times are appended by update function
#     - other indices are fixed, e.g. location, new locations are not allowed, all locations must be provided
#       by the passed DF. However
#        !! Need some mean to trace unused indices, relying on placehloders not possible for general types.
#
#     - 'df' is used ONLY for column names/types (besides 'time_stamp' and 'location').
#     - index_cols, [time_col, ...]
#     - 'max_locations' is the fixed size for the location dimension.
#     - idx_ranges reserved limits for each index column, the first index is just initali limit, could be appended.
#     - Store an empty location_map in dataset attributes.
#     - **kwargs can include chunking or compression settings passed to to_zarr().
#       ... chunking coud either go to ds.chunks(...) or thrugh kwargs['enconding']['time'] = {'chunksizes': (1000, 10)}
#
#     df, index_cols, iterators
#     """
#     # Identify the data columns (excluding time_stamp, location)
#
#     # Build an Xarray Dataset with dims: (time=0, location=max_locations)
#     # We'll create a coordinate for "time" (initially empty),
#     # and a coordinate for "location" (just an integer range 0..max_locations-1).
#
#     ds_coords = [c.make_ds_coord(df[c.struct_cols]) for c in coords] # dim coords
#
#     coord_sizes = [ (col, len(df[col])) for col in index_cols.keys()]
#     append_dim = coord_sizes[0][0]
#     ds = xr.Dataset(
#         coords = dict(coords),
#         attrs = { "coords_valid_size" : dict(coord_sizes),
#                   "append_dim" : append_dim }
#     )
#
#     shape = tuple(ds.dims.values())
#     dim_names = tuple(ds.dims.keys())
#
#
#     # For each data column in df, define an empty variable (0, max_locations)
#     # with a guessed dtype from the Polars column
#     data_cols = [c for c in df.columns if c not in index_cols]
#
#     for col in data_cols:
#         col_dtype, zero, inc_fn = _guess_col_dtype(df[col].dtype)
#         # shape=(0, max_locations)
#         data = np.empty(shape, dtype=col_dtype)
#         ds[col] = (dim_names, data)
#
#     # Write to Zarr
#     ds.to_zarr(str(zarr_path), mode="w", **kwargs)
#     return ds
#
#



# def update(zarr_path: Path, df: pl.DataFrame):
#     """
#     Update the Zarr store created by `create` with new rows of data.
#
#     The function does the following:
#       - Opens the existing dataset.
#       - Determines the dynamic (appendable) dimension from ds.attrs["append_dim"].
#       - For each row in the new DataFrame, finds the proper position along the dynamic dimension:
#           • If the time value exists already, its data variables are overwritten.
#           • If not, the new time is appended (extending the dynamic coordinate and data arrays).
#       - For the fixed index columns (which may be non-struct or struct with multiple coordinate variables),
#         the function matches the row’s value with the valid portion of the coordinate array (using ds.attrs["coords_valid_size"]).
#       - Finally, the updated dataset is written back to the Zarr store.
#
#     Parameters:
#       zarr_path : Path to the Zarr store.
#       df        : Polars DataFrame with the same columns as originally provided to `create`.
#     """
#     # Open the existing Zarr dataset.
#     ds = xr.open_zarr(str(zarr_path))
#     ds = update_xarray_nd(ds, df)
#     # Write the updated dataset back to the Zarr store.
#     ds.to_zarr(str(zarr_path), mode="w")
#     return ds
#
#
# def pivot_nd(df: pl.DataFrame, dims: list[str], value_cols: List[str], unique_coords: Dict[str, np.ndarray], fill_value=np.nan):
#     """
#     Pivot a Polars DataFrame with columns for each dimension in `dims`
#     and one value column `value_col` into an N-dim NumPy array.
#
#     If unique_coords is provided, it is used as the sorted unique values for each dimension.
#
#     Returns a tuple (result, unique_coords) where:
#       - result is an array of shape (n0, n1, ... n_{N-1}) with values from `value_col`
#         (if duplicate keys occur, later values win),
#       - unique_coords is a dict mapping each dimension to its sorted unique values.
#     """
#     idx_list = []
#     for d in dims:
#         # Convert the provided unique coordinates to an array.
#         provided = np.asarray(unique_coords[d])
#         # Sort the provided values (to allow binary search).
#         perm = np.argsort(provided)
#         sorted_coords = provided[perm]
#         # Use searchsorted on the sorted coordinates.
#         idx_sorted = np.searchsorted(sorted_coords, df[d].to_numpy())
#         # Map back to the original ordering.
#         final_idx = perm[idx_sorted]
#         idx_list.append(final_idx)
#     idx_arr = np.vstack(idx_list)
#
#     shape = tuple(len(unique_coords[d]) for d in dims)
#     flat_idx = np.ravel_multi_index(idx_arr, dims=shape)
#
#     def values(column_vals):
#         result = np.full(shape, fill_value,
#                          dtype=column_vals.dtype if np.issubdtype(column_vals.dtype, np.floating) else object)
#         result.flat[flat_idx] = column_vals
#         return result
#
#     output_dict = {
#         var: values(df[var].to_numpy())
#         for var in value_cols
#     }
#     return output_dict
#
#
#
# def intersect_coords(coords, df_coords):
#     mask = np.isin(coords, df_coords)
#     return coords[mask]
#
#
# def process_update_coords(ds: xr.Dataset, df: pl.DataFrame) -> (xr.Dataset, dict):
#     """
#     Process coordinate updates before updating variables.
#
#     - Asserts that the update DF has a column for every coordinate in ds.
#     - For each fixed dimension (all ds.coords except the dynamic dimension,
#       whose name is in ds.attrs["append_dim"]), examines the valid region
#       (ds.coords[d].values[:ds.attrs["coords_valid_size"][d]]) and, if df
#       introduces new values, appends them—provided that the total allocated space
#       is not exceeded.
#     - For the dynamic dimension, extends ds if needed.
#
#     Returns the updated ds and unique_df_coords: a dict mapping each coordinate
#     to the sorted unique values from df.
#     """
#     dims = list(ds.coords.keys())
#     for d in dims:
#         if d not in df.columns:
#             raise ValueError(f"Update DataFrame missing coordinate column '{d}'")
#     dynamic_dim = ds.attrs["append_dim"]
#     fixed_dims = [d for d in dims if d != dynamic_dim]
#
#     unique_df_coords = {
#         d: np.unique(df[d].to_numpy())
#         for d in dims
#     }
#
#     # Process fixed dimensions.
#     for d in fixed_dims:
#         allocated = ds.coords[d].values  # full allocated coordinate array
#         valid_size = ds.attrs["coords_valid_size"][d]  # current valid size
#         ds_valid = allocated[:valid_size]  # currently valid values
#         upd_vals = unique_df_coords[d]  # update DF's unique values for d
#         missing_mask = ~np.isin(upd_vals, ds_valid)
#         missing_vals = upd_vals[missing_mask]
#         if missing_vals.size > 0:
#             new_valid_size = valid_size + missing_vals.size
#             if new_valid_size > len(allocated):
#                 raise ValueError(
#                     f"Not enough allocated space for dimension '{d}': need {new_valid_size}, allocated {len(allocated)}"
#                 )
#             # Append the missing values.
#             new_allocated = allocated.copy()
#             new_allocated[valid_size:new_valid_size] = missing_vals
#             ds = ds.assign_coords({d: new_allocated})
#             ds.attrs["coords_valid_size"][d] = new_valid_size
#     # Process dynamic dimension.
#     upd_dyn = unique_df_coords[dynamic_dim]
#     ds_dyn = ds.coords[dynamic_dim].values
#     missing_dyn = upd_dyn[~np.isin(upd_dyn, ds_dyn)]
#     if missing_dyn.size > 0:
#         extended_dyn = np.concatenate([ds_dyn, missing_dyn])
#         ds = ds.reindex({dynamic_dim: extended_dyn}, fill_value=np.nan)
#     df_coords = {d: intersect_coords(ds.coords[d].values, unique_df_coords[d]) for d in dims}
#     return ds, df_coords
#
#
# def update_xarray_nd(ds: xr.Dataset, df: pl.DataFrame) -> xr.Dataset:
#     """
#     Update an xarray Dataset using an update provided as a Polars DataFrame.
#
#     The DataFrame must have coordinate columns matching ds.coords.
#     For each variable in df that is not a coordinate and exists in ds.data_vars,
#     an update is applied as follows:
#       1. The update coordinates are pre-processed (via process_update_coords) to update
#          fixed dimensions and extend the dynamic dimension if needed.
#       2. For each variable update, pivot the update DF using the unique_df_coords
#          (from process_update_coords) to create an N-dim update array.
#       3. Read the current stored values from ds at the positions given by unique_df_coords,
#          combine them with the update (keeping current values where the update is NaN),
#          and then assign back in one vectorized operation.
#     """
#     dims = list(ds.coords.keys())
#     for d in dims:
#         if d not in df.columns:
#             raise ValueError(f"Update DataFrame missing coordinate column '{d}'")
#
#     # Pre-process the coordinate updates.
#     ds, unique_df_coords = process_update_coords(ds, df)
#
#     # Determine update variable names: those DF columns not in dims and existing in ds.data_vars.
#     update_vars = [col for col in df.columns if (col not in dims) and (col in ds.data_vars)]
#
#     for var in update_vars:
#         update_arr = pivot_nd(df, dims, value_col=var, unique_coords=unique_df_coords)
#         # Read current values from ds for the cells defined by unique_df_coords.
#         current_vals = ds[var].sel(unique_df_coords).values
#         # Where update_arr is not NaN, use that value; otherwise keep current_vals.
#         combined = np.where(np.isnan(update_arr), current_vals, update_arr)
#         ds[var].loc[unique_df_coords] = combined
#     return ds
#
#
# def read(zarr_path: Path, time_stamp_slice, locations):
#     """
#     Read a subset of the data for a given time slice and a list of locations.
#
#     - time_stamp_slice: a tuple (start, end)
#     - locations: list of location codes.
#
#     Returns a Polars DataFrame with columns: time_stamp, location, and each data variable.
#     For each (time_stamp, location) pair in the slice, the stored value is returned.
#     """
#     ds = xr.open_zarr(str(zarr_path), chunks=None)
#     time_col = "time_stamp"
#     loc_col = "location"
#
#     time_coords = ds.coords[time_col].values
#     start, end = time_stamp_slice
#     mask = (time_coords >= start) & (time_coords <= end)
#     sel_time_idx = np.where(mask)[0]
#     sel_times = time_coords[sel_time_idx]
#
#     # Get the location map and select the slots corresponding to requested locations.
#     loc_map = json.loads(ds.attrs.get("location_map", "{}"))
#     selected_locs = {loc: loc_map[str(loc)] for loc in locations if str(loc) in loc_map}
#
#     data_cols = [col for col in ds.data_vars if col not in [time_col, loc_col]]
#     out_time = []
#     out_loc = []
#     out_data = {col: [] for col in data_cols}
#
#     for t_idx in sel_time_idx:
#         for loc, slot in selected_locs.items():
#             out_time.append(time_coords[t_idx])
#             out_loc.append(loc)
#             for col in data_cols:
#                 val = ds[col].values[t_idx, slot]
#                 out_data[col].append(val)
#     ds.close()
#
#     data_dict = {time_col: out_time, loc_col: out_loc}
#     data_dict.update(out_data)
#     return pl.DataFrame(data_dict)