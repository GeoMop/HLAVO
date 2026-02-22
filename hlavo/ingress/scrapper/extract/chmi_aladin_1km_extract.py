import bz2
import yaml
import tempfile

import xarray as xr
import polars as pl

from pathlib import Path

GRIB_TO_SCHEMA = {
    "msl": "air_pressure_at_sea_level",
}


def rename_vars_by_schema(ds: xr.Dataset, schema_vars: dict) -> xr.Dataset:
    allowed_targets = set((schema_vars or {}).keys())
    if not allowed_targets:
        return ds

    rename_map = {}
    for src_name in ds.data_vars:
        tgt_name = GRIB_TO_SCHEMA.get(src_name)
        if not tgt_name or tgt_name not in allowed_targets or tgt_name in ds.data_vars:
            continue

        rename_map[src_name] = tgt_name

    return ds.rename(rename_map) if rename_map else ds


def _read_domain_limits(attrs: dict) -> dict[str, tuple[float, float]]:
    dom_min = attrs.get("meteo_domain_min")
    dom_max = attrs.get("meteo_domain_max")

    if not (isinstance(dom_min, (list, tuple)) and isinstance(dom_max, (list, tuple))):
        raise ValueError("Schema ATTRS must contain meteo_domain_min/max as [lon, lat] lists")

    if not (len(dom_min) == 2 and len(dom_max) == 2):
        raise ValueError("meteo_domain_min/max must have exactly 2 elements: [lon, lat]")

    lon_min, lat_min = dom_min
    lon_max, lat_max = dom_max

    return {
        "longitude": (float(lon_min), float(lon_max)),
        "latitude": (float(lat_min), float(lat_max)),
    }


def cut_to_meteo_domain(ds: xr.Dataset, attrs: dict) -> xr.Dataset:
    limits = _read_domain_limits(attrs or {})
    if not limits:
        return ds

    out = ds

    if "longitude" in limits and "longitude" in out.coords:
        lo, hi = limits["longitude"]
        out = out.sel(longitude=slice(lo, hi))

    if "latitude" in limits and "latitude" in out.coords:
        lo, hi = limits["latitude"]
        lat_vals = out["latitude"].values
        if lat_vals[0] <= lat_vals[-1]:
            out = out.sel(latitude=slice(lo, hi))
        else:
            out = out.sel(latitude=slice(hi, lo))

    return out


def rename_step_to_date_time(ds: xr.Dataset) -> xr.Dataset:
    step_name = "steps" if "steps" in ds.coords else ("step" if "step" in ds.coords else None)
    if step_name is None:
        return ds

    if "time" not in ds.coords:
        return ds.rename({step_name: "date_time"})

    time = ds[step_name] + ds["time"]
    ds = ds.assign_coords(date_time=time)

    if step_name in ds.dims:
        ds = ds.swap_dims({step_name: "date_time"})

    ds = ds.drop_vars(step_name)
    return ds


def transform_grib_ds(ds: xr.Dataset, *, schema: dict, dataset_key: str) -> xr.Dataset:
    if dataset_key not in schema:
        raise KeyError(f"Dataset '{dataset_key}' not found in schema. Keys: {list(schema.keys())}")

    dataset_schema = schema[dataset_key] or {}
    schema_vars = dataset_schema.get("VARS", {}) or {}
    global_attrs = schema.get("ATTRS", {}) or {}

    ds = rename_vars_by_schema(ds, schema_vars)
    ds = cut_to_meteo_domain(ds, global_attrs)
    ds = rename_step_to_date_time(ds)

    return ds


def extractor(payload: bytes, metadata: dict, **kwargs) -> pl.DataFrame | xr.Dataset | bytes | dict | list:
    schema_path = metadata.get("schema_path")
    if not schema_path:
        raise ValueError("metadata.schema_path is missing")

    dataset_name = metadata.get("dataset_name")
    if not dataset_name:
        raise ValueError("metadata.dataset_name is missing")

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

    return transform_grib_ds(ds, schema=schema, dataset_key=dataset_name)
