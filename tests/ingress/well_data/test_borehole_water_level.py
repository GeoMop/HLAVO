#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from hlavo.ingress import well_data
from hlavo.ingress.well_data import well_data_plot

def main(args):
    defaults = ["../hlavo/ingress/well_data/Vrty_souradnice_perforace.xlsx", "List1", "borehole_water_level_out.csv"]
    xls_file, sheetname, csv_output = (args + defaults)[:3]

    final_df = well_data.read_sections_water_levels(xls_file, sheetname)
    print(final_df)

    # reopen and test data
    root_node = well_data._open_zarr_schema(False)
    water_level_node = root_node['Uhelna']['water_levels']
    print(water_level_node.dataset)

    print("--------------------")
    ds = water_level_node.dataset['water_level']
    print(ds)

    print("--------------------")
    df = water_level_node.read_df( var_names=["well_id", "well_in_section_file", "date_time", "water_depth", "water_level"] )
    print(df)

    well_data.csv_output(csv_output, ds.to_dataframe())
    well_data_plot.pdf_plot("borehole_water_level_out.pdf", df, "21")

if __name__ == "__main__":
   main(sys.argv[1:])
