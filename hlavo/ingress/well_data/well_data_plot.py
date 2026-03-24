#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import polars as pl
#import logging

#logger = logging.getLogger(__name__)


def pdf_plot(pdf_file, df, well_id):
    """
    Plots water level data of one well to graph and performs it to PDF file.

    Param   pdf_file   Path to output pdf file
    Param   df         DataFrame containing data from all wells
    Param   well_id    Index of well which data will be performed to graph
    """
    import matplotlib.pyplot as plt

    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)

    full_path = workdir / pdf_file

    df_filtered = df.filter(pl.col("well_id") == well_id)
    ax = df_filtered.plot(
        x="date_time",
        y=["water_depth", "water_level"],
        figsize=(10, 5)
    )

    ax.title("Water levels well: '" + well_id + "'")
    ax.set_xlabel("Date")
    ax.set_ylabel("water level [m]")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()


def main():
    print("Not implemented yet!")

if __name__ == "__main__":
   main()
