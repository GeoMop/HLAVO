#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
import pandas as pd
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

    df_filtered = df[df["well_id"] == well_id]
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
    plt.savefig(pdf_file)
    plt.close()


def main():
    print("Not implemented yet!")

if __name__ == "__main__":
   main()
