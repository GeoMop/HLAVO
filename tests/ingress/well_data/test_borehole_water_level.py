#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
from hlavo.ingress import well_data

def main(args):
    defaults = ["../hlavo/ingress/well_data/Vrty_souradnice_perforace.xlsx", "List1", "borehole_section_out.csv"]
    xls_file, sheetname, csv_output = (args + defaults)[:3]

    final_df = well_data.read_sections_water_levels(xls_file, sheetname)
    print(final_df)

    # reopen and test data
    root_node = well_data._open_zarr_schema(False)
    water_level_node = root_node['Uhelna']['water_levels']
    print(water_level_node.dataset)

    # well_data.csv_output(csv_output, final_df)

if __name__ == "__main__":
   main(sys.argv[1:])
