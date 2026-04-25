#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

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


def test_borehole_draw(tmp_path):
    xls_file = well_data_path / "25_09_27_Odbery_Uhelna.xlsx"
    sheetname = "List1"
    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    section_sheetname = "List1"
    csv_path = "./borehole_water_draw_out.csv"
    store_url = tmp_path / "well_data_store"

    well_data._remove_zarr_store(STORE_URL=store_url)
    df_sections = _sections_with_draw_well(
        well_data.read_sections(section_file, section_sheetname)
    )
    excel_df = well_data.read_draw(xls_file, sheetname, df_sections, STORE_URL=store_url)
    assert not excel_df.empty
    assert (excel_df["well_id"] == "Uh-draw").all()
    assert excel_df["cum_draw"].notna().any()
    assert excel_df["longitude"].notna().all()
    assert excel_df["latitude"].notna().all()
    print(excel_df)
    csv_output(csv_file=csv_path, df=excel_df)
    
    
def test_borehole_water_level(tmp_path):
    # tests of existing files
    assert (well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_nové_vše.xlsx").exists()
    assert (well_data_path / "25_09_27_vrty_staré_vše.xlsx").exists()

    section_file = well_data_path / "Vrty_souradnice_perforace.xlsx"
    sheetname = "List1"
    section_csv = "borehole_water_level_out.csv"
    draw_file = well_data_path / "25_09_27_Odbery_Uhelna.xlsx"
    draw_sheetname = "List1"
    water_level_files = [well_data_path / "25_09_27_vrty_III.etapa_vše.xlsx",
                         well_data_path / "25_09_27_vrty_nové_vše.xlsx",
                         well_data_path / "25_09_27_vrty_staré_vše.xlsx"]
    store_url = tmp_path / "well_data_store"

    well_data._remove_zarr_store(STORE_URL=store_url)
    df_sections = _sections_with_draw_well(well_data.read_sections(section_file, sheetname))
    well_data.read_draw(draw_file, draw_sheetname, df_sections, STORE_URL=store_url)
    final_df = well_data.read_sections_water_levels(
        df_sections,
        water_level_files,
        STORE_URL=store_url,
    )
    print(final_df)

    # reopen and test data
    root_node = well_data._open_zarr_schema(STORE_URL=store_url)
    water_level_node = root_node['Uhelna']['water_levels']
    print(water_level_node.dataset)

    print("--------------------")
    ds = water_level_node.dataset['water_level']
    print(ds)

    print("--------------------")
    df = water_level_node.read_df( var_names=["well_id", "well_in_section_file", "date_time", "water_depth", "water_level"] )
    print(df)
    assert not df.empty
    assert df["water_level"].notna().any()

    for well_id in ["19", "21", "22"]:
        df_well = df.loc[df["well_id"] == well_id]
        assert not df_well.empty
        assert df_well["water_level"].notna().any()

    csv_output(section_csv, ds.to_dataframe())
    water_draw_node = root_node["Uhelna"]["water_draw"]
    df_draw = water_draw_node.read_df(
        var_names=["date", "cum_draw", "well_id", "longitude", "latitude"]
    )
    assert not df_draw.empty
    assert (df_draw["well_id"] == "Uh-draw").all()
    assert df_draw["cum_draw"].notna().any()
    well_data_plot.pdf_plot_all(
        "well_data_test.pdf",
        df_draw=df_draw,
        df_water_levels=df,
        water_level_well_ids=["19", "21", "22"],
    )





if __name__ == "__main__":
   test_borehole_water_level()
   test_borehole_draw()
