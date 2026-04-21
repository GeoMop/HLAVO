#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import pandas as pd
from pathlib import Path
from hlavo.ingress import well_data
from hlavo.ingress.well_data import well_data_plot

well_data_path = Path(well_data.__file__).parent


def csv_output(csv_file, df):
    """
    Perform pandas.DataFrame data to CSV file.
    """
    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)

    full_path = workdir / csv_file
    df.to_csv(path_or_buf=full_path, header=True, mode='w')


def _sections_with_draw_well(df_sections):
    draw_well_id = "Uh-draw"
    assert (df_sections["well_id"] == draw_well_id).any()

    df_draw_section = df_sections.loc[df_sections["well_id"] == draw_well_id]
    assert df_draw_section["longitude"].notna().all()
    assert df_draw_section["latitude"].notna().all()

    return df_sections


def test_borehole_sections():
    xls_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    sheetname = "List1"
    csv_path = "./borehole_section_out.csv"

    assert xls_file.exists()

    excel_df = well_data.read_sections(xls_file, sheetname)
    print(excel_df)
    csv_output(csv_path, excel_df)


def test_borehole_draw():
    xls_file = well_data_path / "25_09_27_Odbery_Uhelna.xlsx"
    sheetname = "List1"
    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    section_sheetname = "List1"
    csv_path = "./borehole_water_draw_out.csv"

    well_data._remove_zarr_store()
    df_sections = _sections_with_draw_well(
        well_data.read_sections(section_file, section_sheetname)
    )
    excel_df = well_data.read_draw(xls_file, sheetname, df_sections)
    print(excel_df)
    csv_output(csv_file=csv_path, df=excel_df)
    
    
def test_borehole_water_level():
    # allow to perform plot of set of wells or single well
    plot_set = True

    # tests of existing files
    assert (well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_nové_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_staré_vše.xlsx").exists()

    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    sheetname = "List1"
    section_csv = "borehole_water_level_out.csv"
    water_level_files = [well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx",
                         well_data_path / "25_09_27_vrty_nové_vše.xlsx",
                         well_data_path / "25_09_27_vrty_staré_vše.xlsx"]

    well_data._remove_zarr_store()
    df_sections = well_data.read_sections(section_file, sheetname)
    final_df = well_data.read_sections_water_levels(df_sections, water_level_files)
    print(final_df)

    # reopen and test data
    root_node = well_data._open_zarr_schema()
    water_level_node = root_node['Uhelna']['water_levels']
    print(water_level_node.dataset)

    print("--------------------")
    ds = water_level_node.dataset['water_level']
    print(ds)

    print("--------------------")
    df = water_level_node.read_df( var_names=["well_id", "well_in_section_file", "date_time", "water_depth", "water_level"] )
    print(df)

    csv_output(section_csv, ds.to_dataframe())
    if plot_set:
        well_data_plot.pdf_plot_multi("water_level", df, {"21", "22", "19"})
    else:
        well_data_plot.pdf_plot_simple("borehole_water_level_out.pdf", df, "21")





if __name__ == "__main__":
   test_borehole_water_level()
   test_borehole_draw()
