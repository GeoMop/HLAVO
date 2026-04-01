#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import logging
import zarr_fuse as zf
from dotenv import load_dotenv
import polars as pl
import matplotlib.pyplot as plt

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


def read_data():
    """
    Read moist data from zarr_fuse storage.

    Returns dataframe contains data of moisture dataset
    """
    root_node = _open_zarr_schema()
    water_level_node = root_node['Uhelna']['profiles']

    df = water_level_node.read_df(
        var_names=["date_time", "site_id", "depth_level", "moisture"])
    return df


def apply_filter(df, site_id, depth_level, out_file):
    """
    IN PROGRESS. Plot data of one side and depth level.

    Param   df            dataframe of moisture data
    Param   site_id       Index of site
    Param   depth_level   Order of depth level
    Returns pandas:DataFrame
    """
    df_pandas = df.to_pandas()
    df_filtered = df_pandas[
        (df_pandas["site_id"] == site_id) &
        (df_pandas["depth_level"] == depth_level)
        ]
    print(df_filtered)
    ax = df_filtered.plot(
        x="date_time",
        y=["moisture"],
        figsize=(10, 5)
    )

    ax.set_title("Moisture data of site: '" + str(site_id) + "', level: '" + str(depth_level) + "'")
    ax.set_xlabel("Date")
    ax.set_ylabel("Moisture")
    ax.grid(True)
    ax.get_figure()

    workdir = _create_work_dir()
    full_path = workdir / out_file

    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()


def main():
    final_df = read_data()
    print(final_df)


if __name__ == "__main__":
   main()
