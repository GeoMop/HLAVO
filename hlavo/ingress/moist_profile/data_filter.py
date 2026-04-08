#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import zarr_fuse as zf
from dotenv import load_dotenv
import xarray as xr
import polars as pl
import matplotlib
matplotlib.use('Agg')  # for interactive graphs

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['hatch.linewidth'] = 6

logger = logging.getLogger(__name__)


def _create_work_dir():
    """
    Create workdir if doesn't exist, return path
    """
    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)
    return workdir


def _open_zarr_schema():
    script_dir = Path(__file__).parent
    root_path = script_dir / "../../.."
    file_path = root_path / ".secrets_env"
    load_dotenv(dotenv_path=file_path)

    schema_path = script_dir / "profile_schema.yaml"
    return zf.open_store(schema_path)


def _filter_jumps(df):
    """
    Input arg of type pandas.DataFrame contains columns 'date_time' and 'moisture'.
    For values in column 'moisture':
     - detect any sudden jump parameterized by size of jump as multiple of “average diff” in  neighbourhood 
     - or detect outliers in the time series of diffs
     - create new variable with data shifted by accumulative jumps and return its
    """
    required_columns = {"date_time", "moisture"}
    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Missing required columns: {sorted(missing_columns)}"

    if len(df) < 3:
        result = df.copy()
        result["moisture_filtered"] = result["moisture"]
        result["moisture_jump"] = False
        result["moisture_correction"] = 0.0
        return result

    ordered = df.sort_values("date_time").copy()
    diff = ordered["moisture"].diff()
    abs_diff = diff.abs()

    neighbourhood_diff = abs_diff.rolling(window=11, center=True, min_periods=3).mean()
    neighbourhood_diff = neighbourhood_diff.fillna(abs_diff.mean())
    neighbourhood_diff = neighbourhood_diff.fillna(0.0)

    diff_center = diff.median(skipna=True)
    diff_mad = (diff - diff_center).abs().median(skipna=True)
    robust_sigma = 1.4826 * diff_mad if pd.notna(diff_mad) else 0.0

    local_threshold = 6.0 * neighbourhood_diff
    global_threshold = 8.0 * robust_sigma
    min_threshold = max(float(abs_diff.median(skipna=True) or 0.0), 1e-9)
    jump_threshold = np.maximum(local_threshold.to_numpy(), max(global_threshold, min_threshold))

    jump_mask = abs_diff.to_numpy() > jump_threshold
    jump_mask[0] = False
    jump_series = pd.Series(jump_mask, index=ordered.index)

    jump_diff = diff.where(jump_series, 0.0).fillna(0.0)
    cumulative_correction = jump_diff.cumsum()

    ordered["moisture_jump"] = jump_series
    ordered["moisture_correction"] = cumulative_correction
    ordered["moisture_filtered"] = ordered["moisture"] - cumulative_correction

    logger.debug(
        "Detected %s moisture jumps for site=%s depth=%s",
        int(jump_series.sum()),
        ordered["site_id"].iloc[0] if "site_id" in ordered.columns and not ordered.empty else None,
        ordered["depth_level"].iloc[0] if "depth_level" in ordered.columns and not ordered.empty else None,
    )

    return ordered.sort_index()


def _plot_single_site_level(df, site_id, depth_level):
    """
    Plot graph of one site_id and depth_level.

    Param   df            dataframe of moisture data
    Param   site_id       Index of site
    Param   depth_level   Order of depth level
    Returns plt object
    """
    df_filtered = df[
        (df["site_id"] == site_id) &
        (df["depth_level"] == depth_level)
        ]
    #print(df_filtered)
    df_filter_jumps = _filter_jumps(df_filtered)
    ax = df_filter_jumps.plot(
        x="date_time",
        y=["moisture", "moisture_filtered"],
        figsize=(10, 5)
    )

    ax.set_title("Moisture data of site: '" + str(site_id) + "', level: '" + str(depth_level) + "'")
    ax.set_xlabel("Date")
    ax.set_ylabel("Moisture")
    ax.grid(True)
    return ax.get_figure()


def read_data(site_ids, depth_levels):
    """
    Read moist data from zarr_fuse storage.

    Param   site_ids      List of indexes of site
    Param   depth_levels  List of orders of depth levels
    Returns xArray.Dataset contains data of moisture dataset
    """
    root_node = _open_zarr_schema()
    water_level_node = root_node['Uhelna']['profiles']

    ds = water_level_node.dataset
    ds["moisture_filtered"] = xr.full_like(ds["moisture"], np.nan)

    for site_id in site_ids:
        for depth_level in depth_levels:
            subset = ds.sel(site_id=site_id, depth_level=depth_level)
            df = subset.to_dataframe().reset_index()

            if df.empty:
                continue

            df_filtered = _filter_jumps(df)
            filtered_values = xr.DataArray(
                df_filtered["moisture_filtered"].to_numpy(),
                dims=subset["moisture"].dims,
                coords=subset["moisture"].coords,
            )
            ds["moisture_filtered"].loc[dict(site_id=site_id, depth_level=depth_level)] = filtered_values

    return ds


def apply_filter(df, site_ids, depth_levels, out_file):
    """
    IN PROGRESS. Plot data of one side and depth level.

    Param   df            dataframe of moisture data
    Param   site_ids      List of indexes of site
    Param   depth_levels  List of orders of depth levels
    Param   out_file      Name of output pdf file
    """
    df_pandas = df.to_pandas()

    workdir = _create_work_dir()
    full_out_file = out_file + ".pdf"
    full_path = workdir / full_out_file
    pdf = PdfPages(full_path)

    for site_id in site_ids:
        for depth_level in depth_levels:
            fig = _plot_single_site_level(df_pandas, site_id, depth_level)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf.close()


def main():
    final_df = read_data()
    print(final_df)


if __name__ == "__main__":
   main()
