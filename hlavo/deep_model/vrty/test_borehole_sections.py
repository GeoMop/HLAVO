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
    # read data from excel, set required column names
    column_map = {
        "Vrt_bez_poradi_vPR": "well_id",
        "x_SJTSK" : "X",
        "y_SJTSK": "Y",
        "ZOB": "Z",
        "Hloubka": "depth",
        "Kolektor_puv": "collector",
        "Perf_dilci" : "interval_A",
        "Z_OD": "Z_OD",
        "Z_DO": "Z_DO",
        "OD": "OD",
        "DO": "DO"
    }
    df = pd.read_excel(io=xls_file, sheet_name=sheetname, header=0, usecols=column_map.keys())
    df = df.rename(columns=column_map)

    # split more intervals to separate rows
    df["interval_A"] = df["interval_A"].str.split(";")
    df = df.explode("interval_A", ignore_index=True)

    tmp = df["interval_A"].str.extract(
        r"(?P<interval_A_min>\d+(?:\.\d+)?)\s*-\s*(?P<interval_A_max>\d+(?:\.\d+)?)"
    )
    df["interval_A_max"] = tmp["interval_A_max"].astype(float)
    df["interval_A_min"] = tmp["interval_A_min"].astype(float)

    # add column interval_num_from_top (numbering of rows with same well_id)
    df["interval_num_from_top"] = df.groupby(["well_id", "collector"]).cumcount()

    # check values, print different problems
    invalid_mask_interval = df["interval_A_min"] >= df["interval_A_max"]
    invalid_rows_interval = df[invalid_mask_interval]
    for idx in invalid_rows_interval.index:
        print(f"Invalid interval at row {idx}: min={df.at[idx, 'interval_A_min']}, max={df.at[idx, 'interval_A_max']}")

    invalid_mask_from = df["Z_OD"] == df["Z"] - df["DO"]
    invalid_rows_from = df[invalid_mask_from]
    for idx in invalid_rows_from.index:
        print(f"Invalid \'Z_OD\' value at row {idx}: Z_OD={df.at[idx, 'Z_OD']}, Z={df.at[idx, 'Z']}, DO={df.at[idx, 'DO']}. It should be \'Z_OD = Z - DO\'")

    invalid_mask_to = df["Z_DO"] == df["Z"] - df["OD"]
    invalid_rows_to = df[invalid_mask_to]
    for idx in invalid_rows_to.index:
        print(f"Invalid \'Z_DO\' value at row {idx}: Z_DO={df.at[idx, 'Z_DO']}, Z={df.at[idx, 'Z']}, OD={df.at[idx, 'OD']}. It should be \'Z_DO = Z - OD\'")

    expected_from = df.groupby(["well_id", "collector"])["interval_A_min"].transform("min")
    expected_to = df.groupby(["well_id", "collector"])["interval_A_max"].transform("max")
    invalid_mask_interval = (df["OD"] != expected_from) | (df["DO"] != expected_to)
    for idx, row in df.loc[invalid_mask_interval].iterrows():
        print(f"Invalid \'OD - DO\' interval at row {idx}: OD={row['OD']}, expected={expected_from.loc[idx]}; DO={row['DO']}, expected={expected_to.loc[idx]}")

    # remove unnecessary columns
    df = df.drop(columns=["Z_OD", "Z_DO", "OD", "DO", "interval_A"])

    return df

"""
Perform data to CSV file.
"""
def csv_output(csv_file, df):
    df.to_csv(path_or_buf=csv_file, header=True, mode='w')
    # see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv


def main(argv):
    if (len(argv) != 3):
        argv = ["./Vrty_souradnice_perforace.xlsx", "List1", "Vrty_uprav.csv"]
        # temporary hack, set args automatically if they are not set
        #print("Invalid number of input args, must be 3! ")
        #sys.exit(1)

    excel_df = load_data_from_excel(xls_file=argv[0], sheetname=argv[1])
    print(excel_df)
    ## do something
    csv_output(csv_file=argv[2], df=excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])
