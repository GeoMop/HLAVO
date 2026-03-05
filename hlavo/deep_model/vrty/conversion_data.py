#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


"""
Process data of one sheet of excel file.
"""
def _process_water_level_sheet(df_read, sheetname):
    # read data from excel, set required column names
    df_read = df_read.rename(columns={
        df_read.columns[1]: "date_time",
        "FINAL záměr (m)": "water_depth",
        "FINAL hladina (m nm)": "water_level"
    })
    df_read["well_id_orig"] = sheetname
    df_read["well_id"] = df_read["well_id_orig"].values.astype("str")
    df = df_read[["well_id", "date_time", "water_depth", "water_level"]].sort_values("date_time").reset_index(drop=True)

    # remove rows contains NaN values of water_level
    df = df.dropna(subset=["water_level"])

    # convert values of date_time column to datetime64[min]
    df["date_time"] = pd.to_datetime(df["date_time"]).dt.floor("min")

    df.attrs["units"] = {"water_depth ": "m", "water_level": "m above see level"}

    return df

"""
Load data from list of excel files (*.xls, *.xlsx ...).
"""
def read_water_level(file_paths=None):
    # use default input files  if file_paths is not set
    if file_paths is None:
        defautl_files = ["./25_09_27_vrty_III.etapa_vše.xlsx", "./25_09_27_vrty_nové_vše.xlsx",
                         "./25_09_27_vrty_staré_vše.xlsx"]
        script_path = Path(__file__).resolve().parent
        file_paths = {script_path / f for f in defautl_files}

    # List of DataFrames of sheets with required data format
    dfs = []

    for xls_file in file_paths:
        print(f"Processing of file: {xls_file}")
        # List of file sheets
        xls = pd.ExcelFile(xls_file)

        for sheet in xls.sheet_names:
            print(f" Reading of sheet: {sheet}")
            df = pd.read_excel(xls, sheet_name=sheet)
            clmns = df.columns.values.tolist()

            if (clmns[0] == "m nm OB :") and (clmns[2] == "záměr (m)"):
                print("  ... processing")
                df_sheet = _process_water_level_sheet(df, sheetname=sheet)
                dfs.append(df_sheet)
            else:
                print("  ... sheet is not in required data format")

    if dfs:  # test empty list
        final_df = pd.concat(dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df

"""
Perform graph to PDF file.
"""
def pdf_plot(pdf_file, df):
    ax = df.plot(
        x="date_time",
        y=["water_depth", "water_level"],
        figsize=(10, 5)
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("water level [m]")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(pdf_file)
    plt.close()


def main(argv):
    final_df = read_water_level()
    print(final_df)
    #pdf_plot(pdf_file=argv[2], df=final_df)

if __name__ == "__main__":
   main(sys.argv[1:])
