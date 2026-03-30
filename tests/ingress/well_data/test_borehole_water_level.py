#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from hlavo.ingress import well_data
from hlavo.ingress.well_data import well_data_plot

def test_borehole_water_level():
    # allow to perform plot of set of wells or single well
    plot_set = True

    xls_file = "../hlavo/ingress/well_data/Vrty_souradnice_perforace.xlsx"
    sheetname = "List1"
    csv_output = "borehole_water_level_out.csv"

    final_df = well_data.read_sections_water_levels(xls_file, sheetname)
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

    well_data.csv_output(csv_output, ds.to_dataframe())
    if plot_set:
        well_data_plot.pdf_plot_multi("water_level", df, {"21", "22", "19"})
    else:
        well_data_plot.pdf_plot_simple("borehole_water_level_out.pdf", df, "21")

if __name__ == "__main__":
   test_borehole_water_level()
