#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def _process_water_level_sheet(df_read, sheetname):
    """
    Process data of one sheet of excel file.
    """

    # read data from excel, set required column names
    df_read = df_read.rename(columns={
        df_read.columns[1]: "date_time",
        "FINAL záměr (m)": "water_depth",
        "FINAL hladina (m nm)": "water_level"
    })

    # convert values of date_time column to datetime64[min]
    df_read["date_time"] = pd.to_datetime(df_read["date_time"]).dt.floor("min")

    df_read["well_id_orig"] = sheetname
    df_read["well_id"] = df_read["well_id_orig"].values.astype("str")
    df = df_read[["well_id", "date_time", "water_depth", "water_level"]].sort_values("date_time").reset_index(drop=True)

    # remove rows contains NaN values of water_level
    df = df.dropna(subset=["water_level"])

    df.attrs["units"] = {"water_depth ": "m", "water_level": "m above see level"}

    return df


def _prepare_default_filepaths(file_paths):
    """
    Set default paths to input files if file_paths is not set.
    """
    if file_paths is None:
        defautl_files = ["./25_09_27_vrty_III.etapa_vše.xlsx", "./25_09_27_vrty_nové_vše.xlsx",
                         "./25_09_27_vrty_staré_vše.xlsx"]
        script_path = Path(__file__).resolve().parent
        file_paths = {script_path / f for f in defautl_files}
    return file_paths


def _sheet_names_dictionary():
    """
    Return dictionary of pairs of sheet names in excel files and its according well ids in section file.
    """
    dict = {
        "1420_19 (Mw)" : "19",
        "1420_20 (Nd)" : "20",
        "1420_21 (Mw)" : "21",
        "1420_22 (Mw)" : "22",
        "1420_22B (Q)" : "22B" ,
        "1420_23 (Mw)" : "23",
        "1420_23B (Q)" : "23B",
        "1420_24 (Pw)" : "24",
        "1420_24B (Q)" : "24B",
        "1420_25 (Nd)" : "25",
        "1420_1 (Nd)" : "1",
        "1420_2 (Nd)" : "2",
        "1420_3 (Nd)" : "3",
        "1420_4 (Ng)" : "4",
        "1420_5 (Nd)" : "5",
        "1420_6 (Nd)" : "6",
        "1420_7 (Ng)" : "7",
        "6413_8 (Krystalinikum)" : "8",
        "1430_9 (Q)" : "9",
        "1430_10 (Nd)" : "10",
        "1430_10A (Nd)" : "10A",
        "1430_11 (Nd)" : "11",
        "1430_13 (Nd)" : "13",
        "1430_15 (Nd)" : "15",
        "1430_16 (Nd)" : "16",
        "1420_17 (Nd)" : "17",
        "1420_18 (Nd)" : "18",
        "H-3 (Pw)" : "H-3",
        "H-4 (Pw)" : "H-4",
        "H-6 (Pw)" : "H-6",
        "H-9 (Pw)" : "H-9",
        "H-2a (Mw)" : "H-2a",
        "H-4a (Mw)" : "H-4a",
        "H-7a (Mw)" : "H-7a",
        "H-8a (Mw)" : "H-8a",
        "H-3b (Nd)" : "H-3b",
        "H-5b (Nd)" : "H-5b",
        "H-6b (Nd)" : "H-6b",
        "GI-1 (Q)" : "GI-1",
        "GI-2 (Q)" : "GI-2",
        "GI-3 (Q)" : "GI-3",
        "JA-1 (Nd)" : "JA-1",
        "Uh-2 (Q)" : "UH-2"
    }
    return dict


def read_water_level(file_paths=None):
    """
    Process water level data of set of wells and store them into DataFrame.
    Data is stored in set of excel files (*.xls, *.xlsx ...), each file contains
    set of sheets of well data.

    Param   file_paths    Set of paths to Excel input files
    Returns pandas:DataFrame
    """

    logging.basicConfig(level=logging.INFO)

    file_paths = _prepare_default_filepaths(file_paths)

    # List of DataFrames of sheets with required data format
    dfs = []
    # List of processed sheets in all processed files
    processed_sheets = []

    for xls_file in file_paths:
        logger.info("Processing of file: %s", xls_file)
        # List of file sheets
        xls = pd.ExcelFile(xls_file)

        for sheet in xls.sheet_names:
            logger.info(" Reading of sheet: %s", sheet)
            df = pd.read_excel(xls, sheet_name=sheet)
            clmns = df.columns.values.tolist()

            try:
                df_sheet = _process_water_level_sheet(df, sheetname=sheet)
            except Exception as e:
                logger.exception("message")
            else:
                dfs.append(df_sheet)
                processed_sheets.append(sheet)
                logger.info("  ... sheet successfully processed")

    #processed_sheets = processed_sheets.values.astype("str")
    full_name_map = _sheet_names_dictionary();
    # test if all values of full_name_map dictionary exist as sheet in set of excel files
    expected = set(full_name_map.values())
    missing = expected - set(processed_sheets)
    # TypeError: unsupported operand type(s) for -: 'set' and 'list'
    if missing:
        warn_msg = "Following sheets were not processed or doesn't exist\n"
        for name in sorted(missing):
            warn_msg = warn_msg + " - " + name + "\n"
        logger.warning("%s", warn_msg)

    if dfs:  # test empty list
        final_df = pd.concat(dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df


def read_draw(xls_file, sheetname):
    """
    Process water draw data of one well and store them to DataFrame
    Data is stored in excel file.

    Param   xls_file    Path to Excel input file
    Param   sheetname   Name of sheet in xls_file
    Returns pandas:DataFrame
    """

    logging.basicConfig(level=logging.INFO)
    logger.info("Processing draw data from file %s", xls_file)
    #read data from excel, set required column names
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

    # convert to base unit (m^3)
    df_result["cum_draw"] = df_result["cum_draw"] * 1000

    df_result.attrs["units"] = {"cum_draw ": "m^3"}
    logger.info(" ... draw data completely processed")

    return df_result


def read_sections(section_file, sheetname):
    """
    Process section data of set of well from excel file and store them
    to DataFrame.

    Param   xls_file    Path to Excel input file
    Param   sheetname   Name of sheet in xls_file
    Returns pandas:DataFrame
    """

    logging.basicConfig(level=logging.INFO)

    # read data from excel, set required column names
    column_map = {
        "Vrt_s_kolektorem": "borehole_full_name",
        "Vrt_bez_poradi_vPR": "well_id",
        "x_SJTSK" : "X",
        "y_SJTSK": "Y",
        "ZOB": "Z",
        "Hloubka": "depth",
        "Kolektor_puv": "collector",
        "Perf_dilci" : "interval",
        "Z_OD": "Z_OD",
        "Z_DO": "Z_DO",
        "OD": "OD",
        "DO": "DO"
    }
    df = pd.read_excel(io=section_file, sheet_name=sheetname, header=0, usecols=column_map.keys())
    df = df.rename(columns=column_map)

    # add borehole_id - according name of sheet in excel file if exists
    full_name_map = _sheet_names_dictionary();
    df["borehole_id"] = df["borehole_full_name"].map(full_name_map)
    df["confirmed"] = df["borehole_id"].notna().astype(int)

    # test if all values of full_name_map dictionary exist in dataframe
    expected = set(full_name_map.values())
    used = set(df["borehole_id"].dropna())
    missing = expected - used
    if missing:
        warn_msg = "Following sheet names are not contained in section list\n"
        for name in sorted(missing):
            warn_msg = warn_msg + " - " + name + "\n"
        logger.warning("%s", warn_msg)

    # split more intervals to separate rows
    df["interval"] = df["interval"].str.split(";")
    df = df.explode("interval", ignore_index=True)

    tmp = df["interval"].str.extract(
        r"(?P<interval_min>\d+(?:\.\d+)?)\s*-\s*(?P<interval_max>\d+(?:\.\d+)?)"
    )
    df["interval_max"] = tmp["interval_max"].astype(float)
    df["interval_min"] = tmp["interval_min"].astype(float)

    # add column interval_num_from_top (numbering of rows with same well_id)
    df["interval_num_from_top"] = df.groupby(["well_id", "collector"]).cumcount()

    # check values, print different problems
    invalid_mask_interval = df["interval_min"] >= df["interval_max"]
    invalid_rows_interval = df[invalid_mask_interval]
    for idx in invalid_rows_interval.index:
        logger.warning("Invalid interval at row %s: min=%s, max=%s", idx, df.at[idx, 'interval_min'], df.at[idx, 'interval_max'])

    invalid_mask_from = df["Z_OD"] == df["Z"] - df["DO"]
    invalid_rows_from = df[invalid_mask_from]
    for idx in invalid_rows_from.index:
        logger.warning("Invalid \'Z_OD\' value at row %s: Z_OD=%s, Z=%s, DO=%s. It should be \'Z_OD = Z - DO\'",
                       idx, df.at[idx, 'Z_OD'], df.at[idx, 'Z'], df.at[idx, 'DO'])

    invalid_mask_to = df["Z_DO"] == df["Z"] - df["OD"]
    invalid_rows_to = df[invalid_mask_to]
    for idx in invalid_rows_to.index:
        logger.warning("Invalid \'Z_DO\' value at row %s: Z_DO=%s, Z=%s, OD=%s. It should be \'Z_DO = Z - OD\'",
                       idx, df.at[idx, 'Z_DO'], df.at[idx, 'Z'], df.at[idx, 'OD'])

    expected_from = df.groupby(["well_id", "collector"])["interval_min"].transform("min")
    expected_to = df.groupby(["well_id", "collector"])["interval_max"].transform("max")
    invalid_mask_interval = (df["OD"] != expected_from) | (df["DO"] != expected_to)
    for idx, row in df.loc[invalid_mask_interval].iterrows():
        logger.warning("Invalid \'OD - DO\' interval at row %s: OD=%s, expected=%s; DO=%s, expected=%s",
                       idx, row['OD'], expected_from.loc[idx], row['DO'], expected_to.loc[idx])

    # remove unnecessary columns
    df = df.drop(columns=["Z_OD", "Z_DO", "OD", "DO", "interval"])

    df.attrs["units"] = { "X": "m", "Y": "m", "Z": "m", "depth": "m", "interval_max": "m", "interval_min": "m"}

    return df

def csv_output(csv_file, df):
    """
    Perform pandas.DataFrame data to CSV file.
    """
    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)

    full_path = workdir / csv_file
    df.to_csv(path_or_buf=full_path, header=True, mode='w')


def pdf_plot(pdf_file, df):
    """
    Perform graph to PDF file.
    TODO rename and add documentation
    """
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


def main():
    final_df = read_water_level()
    print(final_df)

if __name__ == "__main__":
   main()
