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
        "MVM1" : "M1",
        "MVM2" : "M2",
        "MVM3" : "M3",
        "MVM4" : "M4",
        "MVM5" : "M5",
        "MVM6" : "M6",
        "MVM7" : "M7",
        "MVM8" : "M8",
        "MVM9" : "M9",
        "MVM10" : "M10",
        "MVM11" : "M11",
        "MVM12" : "M12",
        "ROK" : "year"
    }
    df_long_cols = pd.read_excel(io=xls_file, sheet_name=sheetname, header=0, usecols=column_map.keys())
    df_long_cols = df_long_cols.rename(columns=column_map)

    # transform data, create separate row for each month
    df_months = df_long_cols.melt(
        id_vars="year",
        value_vars=[f"M{i}" for i in range(1, 13)],
        var_name="month",
        value_name="cum_draw"
    )
    df_months["month"] = df_months["month"].str[1:].astype(int)
    df_months["date"] = (
        pd.to_datetime(
            dict(year=df_months["year"], month=df_months["month"], day=28)
        )
    ).values.astype("datetime64[D]")

    # filter output columns to result dataframe
    df_result = df_months[["date", "cum_draw"]].sort_values("date").reset_index(drop=True)

    return df_result

"""
Perform data to CSV file.
"""
def csv_output(csv_file, df):
    df.to_csv(path_or_buf=csv_file, header=True, mode='w')


def main(argv):
    if (len(argv) != 3):
        argv = ["./25_09_27_Odbery_Uhelna.xlsx", "List1", "Odbery_uprav.csv"]
        # temporary hack, set args automatically if they are not set
        #print("Invalid number of input args, must be 3! ")
        #sys.exit(1)

    excel_df = load_data_from_excel(xls_file=argv[0], sheetname=argv[1])
    print(excel_df)
    csv_output(csv_file=argv[2], df=excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])
