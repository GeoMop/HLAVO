import bz2
import yaml
import tempfile
import numpy as np

import xarray as xr
import polars as pl
import zarr_fuse as zf

from pathlib import Path

def rename_vars_by_schema(ds: xr.Dataset, metadata: dict, dataset_key: str, schema_path: Path) -> xr.Dataset:
    data_vars = list(ds.data_vars)
    assert len(data_vars) == 1, f"Expected exactly one data variable in the dataset, found: {data_vars}"

    src_var_in_ds = data_vars[0]

    source_name = (metadata.get("dataframe_row") or {}).get("quantity")
    assert source_name is not None, "metadata.dataframe_row.quantity is missing"

    root = zf.schema.deserialize(schema_path)

    node = root.groups[dataset_key].ds
    source_map = node.source_to_schema_name_map

    target_name = source_map.get(source_name)
    assert target_name is not None, f"Source variable name '{source_name}' not found in schema mapping for dataset '{dataset_key}'"

    if src_var_in_ds == target_name:
        return ds

    assert target_name not in ds.data_vars and target_name not in ds.coords, (
        f"Target variable name '{target_name}' already exists in dataset"
    )

    return ds.rename({src_var_in_ds: target_name})


def cut_to_meteo_domain(ds: xr.Dataset, attrs: dict) -> xr.Dataset:
    dom_min = attrs.get("meteo_domain_min")
    dom_max = attrs.get("meteo_domain_max")

    assert dom_min is not None and dom_max is not None, "Missing meteo_domain_min/meteo_domain_max in schema ATTRS"
    assert len(dom_min) == 2 and len(dom_max) == 2, "meteo_domain_min/max must be [lon, lat]"

    lon_min, lat_min = dom_min
    lon_max, lat_max = dom_max

    lon_vals = ds["longitude"].values
    lat_vals = ds["latitude"].values

    lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
    lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)

    return ds.sel(longitude=lon_slice, latitude=lat_slice)


def rename_step_to_date_time(ds: xr.Dataset) -> xr.Dataset:
    step_name = "step" if "step" in ds.coords else None
    assert step_name is not None, "Expected 'step' coordinate in dataset, but not found"

    time = ds[step_name] + ds["time"]
    ds = ds.assign_coords(date_time=time)

    if step_name in ds.dims:
        ds = ds.swap_dims({step_name: "date_time"})

    ds = ds.drop_vars(step_name)
    ds = ds.drop_vars("time")
    return ds


def drop_unused_vars(ds: xr.Dataset, schema: dict, dataset_key: str) -> xr.Dataset:
    ds_schema = schema.get(dataset_key) or {}
    assert ds_schema, f"Dataset '{dataset_key}' not found in schema. Keys: {list(schema.keys())}"

    vars_map = ds_schema.get("VARS") or {}
    expected_vars = set(vars_map.keys())
    for vdef in vars_map.values():
        df_col = (vdef or {}).get("df_col")
        if df_col:
            expected_vars.add(str(df_col).strip())

    vars_to_drop = [var for var in ds.data_vars if var not in expected_vars]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    expected_coords = set((ds_schema.get("COORDS") or {}).keys())
    coords_to_drop = [coord for coord in ds.coords if coord not in expected_coords and coord not in ds.dims]
    if coords_to_drop:
        ds = ds.drop_vars(coords_to_drop)

    return ds


def add_missing_schema_vars_as_nan(ds: xr.Dataset, dataset_key: str, schema_path: Path) -> xr.Dataset:
    root = zf.schema.deserialize(schema_path)
    node = root.groups[dataset_key].ds

    for var_name, var_def in node.VARS.items():
        if var_name in ds.data_vars:
            continue

        var_coords = tuple(str(coord) for coord in var_def.coords)

        missing_coords = [coord for coord in var_coords if coord not in ds.coords]
        assert not missing_coords, (
            f"Cannot create missing variable '{var_name}', missing coords in dataset: {missing_coords}"
        )

        shape = tuple(ds.sizes[coord] for coord in var_coords)
        data = np.full(shape, np.nan, dtype=float)

        var_attrs = {
            "unit": str(var_def.unit) if getattr(var_def, "unit", None) is not None else None,
            "description": getattr(var_def, "description", None),
            "df_col": getattr(var_def, "df_col", var_name),
        }
        var_attrs = {k: v for k, v in var_attrs.items() if v is not None}

        ds[var_name] = xr.DataArray(
            data=data,
            dims=var_coords,
            coords={coord: ds.coords[coord] for coord in var_coords},
            attrs=var_attrs,
        )

    return ds

def transform_grib_ds(ds: xr.Dataset, *, schema: dict, dataset_key: str, metadata: dict, schema_path: Path) -> xr.Dataset:
    if dataset_key not in schema:
        raise KeyError(f"Dataset '{dataset_key}' not found in schema. Keys: {list(schema.keys())}")

    global_attrs = schema.get("ATTRS", {}) or {}

    ds = rename_vars_by_schema(ds, metadata, dataset_key, schema_path)
    ds = cut_to_meteo_domain(ds, global_attrs)
    ds = rename_step_to_date_time(ds)
    ds = drop_unused_vars(ds, schema, dataset_key)
    ds = add_missing_schema_vars_as_nan(ds, dataset_key, schema_path)

    return ds


def extractor(payload: bytes, metadata: dict, **kwargs) -> pl.DataFrame | xr.Dataset | bytes | dict | list:
    schema_path = metadata.get("schema_path")
    if not schema_path:
        raise ValueError("metadata.schema_path is missing")

    target_node = metadata.get("target_node")
    if not target_node:
        raise ValueError("metadata.target_node is missing")

    parent_dir = Path(__file__).parent.parent
    schema_path = parent_dir / schema_path

    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))

    ct = (metadata.get("content_type") or "").lower()
    is_bz2 = ("bz2" in ct) or str(metadata.get("source_path", "")).endswith(".bz2")
    data = bz2.decompress(payload) if is_bz2 else payload

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "input.grib"
        p.write_bytes(data)
        ds = xr.open_dataset(p, engine="cfgrib").load()

    return transform_grib_ds(ds, schema=schema, dataset_key=target_node, metadata=metadata, schema_path=schema_path)
