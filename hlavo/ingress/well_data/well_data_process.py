#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import argparse
from pathlib import Path
import pandas as pd
from hlavo.ingress import well_data


"""
This script reads, processees and stores cumulation draw and water level data to zarr storeage. 
"""


well_data_path = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot",
        nargs="?",
        default=None,
        choices=["plot"],
        help="Read written data from zarr and generate a single PDF with draw and water-level plots.",
    )
    return parser.parse_args()

def read_borehole_draw(df_sections):
    xls_file = well_data_path / "25_09_27_Odbery_Uhelna.xlsx"
    sheetname = "List1"
    well_data.read_draw(xls_file, sheetname, df_sections)


def read_borehole_water_level(df_sections):
    # tests of existing files
    assert (well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_nové_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_staré_vše.xlsx").exists()

    water_level_files = [well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx",
                         well_data_path / "25_09_27_vrty_nové_vše.xlsx",
                         well_data_path / "25_09_27_vrty_staré_vše.xlsx"]
    well_data.read_sections_water_levels(df_sections, water_level_files)


def _read_optional_df(node, var_names):
    try:
        return node.read_df(var_names=var_names)
    except KeyError:
        return pd.DataFrame(columns=var_names)


def plot_written_data():
    root_node = well_data._open_zarr_schema()
    water_level_node = root_node["Uhelna"]["water_levels"]
    df_water_levels = _read_optional_df(
        water_level_node,
        var_names=["well_id", "well_in_section_file", "date_time", "water_depth", "water_level"]
    )
    water_draw_node = root_node["Uhelna"]["water_draw"]
    df_draw = _read_optional_df(
        water_draw_node,
        var_names=["date", "cum_draw", "well_id", "longitude", "latitude"]
    )
    well_data.pdf_plot_all(
        "well_data.pdf",
        df_draw=df_draw,
        df_water_levels=df_water_levels,
        water_level_well_ids=["19", "21", "22"],
    )


def main():
    args = parse_args()
    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    section_sheetname = "List1"
    df_sections = well_data.read_sections(section_file, section_sheetname)

    well_data._remove_zarr_store()
    read_borehole_draw(df_sections)
    read_borehole_water_level(df_sections)

    if args.plot == "plot":
        plot_written_data()

if __name__ == "__main__":
   main()
