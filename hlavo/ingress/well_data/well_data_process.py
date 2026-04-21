#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

from pathlib import Path
from hlavo.ingress import well_data


"""
This script reads, processees and stores cumulation draw and water level data to zarr storeage. 
"""


well_data_path = Path(__file__).parent

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

def main():
    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    section_sheetname = "List1"
    df_sections = well_data.read_sections(section_file, section_sheetname)

    well_data._remove_zarr_store()
    read_borehole_draw(df_sections)
    read_borehole_water_level(df_sections)

if __name__ == "__main__":
   main()
