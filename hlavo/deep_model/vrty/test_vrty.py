#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

"""
Args:
    xls_file:
    sheetname:
    csv_file:
"""


import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Load data from excel file (*.xls, *.xlsx ...).
"""
def load_data_from_excel(xls_file, sheetname):
    # read data from excel, set required column names
    df_read = pd.read_excel(xls_file, sheet_name=sheetname, header=0)
    df_read = df_read.rename(columns={
        df_read.columns[1]: "date_time",
        "FINAL záměr (m)": "water_intention_(m)",
        "FINAL hladina (m nm)": "water_level_(masl)"
    })
    df_read["well_id_orig"] = sheetname
    df_read["well_id"] = df_read["well_id_orig"].values.astype("str")
    df = df_read[["well_id", "date_time", "water_intention_(m)", "water_level_(masl)"]].sort_values("date_time").reset_index(drop=True)

    # remove rows contains NaN values of water_level
    df = df.dropna(subset=["water_level_(masl)"])

    # convert values of date_time column to datetime64[min]
    df["date_time"] = pd.to_datetime(df["date_time"]).dt.floor("min")

    return df

"""
Perform graph to PDF file.
"""
def pdf_plot(pdf_file, df):
    ax = df.plot(
        x="date_time",
        y=["water_intention_(m)", "water_level_(masl)"],
        figsize=(10, 5)
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("water level [m]")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(pdf_file)
    plt.close()


def main(argv):
    if (len(argv) != 3):
        argv = ["./25_09_27_vrty_III.etapa_vše.xlsx", "19", "water_level_plot.pdf"]
        # temporary hack, set args automatically if they are not set
        #print("Invalid number of input args, must be 3! ")
        #sys.exit(1)

    excel_df = load_data_from_excel(xls_file=argv[0], sheetname=argv[1])
    print(excel_df)
    pdf_plot(pdf_file=argv[2], df=excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])
