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


"""
Load data from excel file (*.xls, *.xlsx ...).
"""
def load_data_from_excel(xls_file, sheetname):
    column_map = {
        "Vrt_bez_poradi_vPR": "well_id",
        "x_SJTSK" : "X",
        "y_SJTSK": "Y",
        "ZOB": "Z",
        "Hloubka": "depth",
        "Kolektor_puv": "collector",
        "Perf_dilci" : "interval_A"
    }
    df = pd.read_excel(io=xls_file, sheet_name=sheetname, header=0, usecols=column_map.keys())
    df = df.rename(columns=column_map)

    df["interval_A"] = df["interval_A"].str.split(";")
    df = df.explode("interval_A", ignore_index=True)

    tmp = df["interval_A"].str.extract(
        r"(?P<interval_A_min>\d+(?:\.\d+)?)\s*-\s*(?P<interval_A_max>\d+(?:\.\d+)?)"
    )
    df["interval_A_max"] = tmp["interval_A_max"].astype(float)
    df["interval_A_min"] = tmp["interval_A_min"].astype(float)
    df = df.drop(columns="interval_A")

    df["interval_num_from_top"] = df.groupby("well_id").cumcount()

    invalid_mask = df["interval_A_min"] >= df["interval_A_max"]
    invalid_rows = df[invalid_mask]
    for idx in invalid_rows.index:
        print("Invalid interval in row {idx}: min={df.at[idx, 'interval_A_min']}, max={df.at[idx, 'interval_A_max']}")

    # see https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas.read_excel
    return df

"""
Perform data to CSV file.
"""
def csv_output(csv_file, df):
    df.to_csv(path_or_buf=csv_file, header=True, mode='w')
    # see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv


def main(argv):
    if (len(argv) != 3):
        print("Invalid number of input args, must be 3! ")
        sys.exit(1)

    excel_df = load_data_from_excel(xls_file=argv[0], sheetname=argv[1])
    print(excel_df)
    ## do something
    csv_output(csv_file=argv[2], df=excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])