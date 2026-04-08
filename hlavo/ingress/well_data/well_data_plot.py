#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')  # for interactive graphs

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['hatch.linewidth'] = 6


#import logging

#logger = logging.getLogger(__name__)


def _create_work_dir():
    """
    Create workdir if doesn't exist, return path
    """
    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)
    return workdir


def _plot_by_well_id(df, well_id):
    """
    Helper function. Prepares plot of water level data of one wel.

    Param   df         DataFrame containing data from all wells
    Param   well_id    Index of well which data will be performed to graph
    """
    df_filtered = df.filter(pl.col("well_id") == well_id)
    df_filtered = df_filtered.to_pandas()
    ax = df_filtered.plot(
        x="date_time",
        y=["water_level"],
        figsize=(10, 5)
    )

    ax.set_title("Water levels well: '" + well_id + "'")
    ax.set_xlabel("Date")
    ax.set_ylabel("water level [m]")
    ax.grid(True)
    return ax.get_figure()


def pdf_plot_simple(pdf_file, df, well_id):
    """
    Plots water level data of one well to graph and performs it to PDF file.

    Param   pdf_file   Path to output pdf file
    Param   df         DataFrame containing data from all wells
    Param   well_id    Index of well which data will be performed to graph
    """
    workdir = _create_work_dir()
    full_path = workdir / pdf_file

    _plot_by_well_id(df, well_id)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()


def pdf_plot_multi(out_file, df, well_ids):
    """
    Plots water level data of one well to graph and performs it to PDF file.

    Param   pdf_file   Path to output pdf file
    Param   df         DataFrame containing data from all wells
    Param   well_ids   List of indexes of well which data will be performed to graphs
    """
    workdir = _create_work_dir()
    pdf_file = out_file + ".pdf"
    full_pdf_path = workdir / pdf_file

    pdf = PdfPages(full_pdf_path)

    for well_id in well_ids:
        fig = _plot_by_well_id(df, well_id)
        fname = out_file + "_" + well_id + ".png"
        f_full_path =  workdir / fname
        fig.savefig(fname=f_full_path, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf.close()


def main():
    print("Not implemented yet!")

if __name__ == "__main__":
   main()
