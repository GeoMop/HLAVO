#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
sys.path.append('../../../hlavo/deep_model/vrty')
import well_data
# from hlavo.deep_model.vrty import well_data


def main(args):
    defaults = ["../../../hlavo/deep_model/vrty/Vrty_souradnice_perforace.xlsx", "List1", "borehole_section_out.csv"]
    xls_file, sheetname, csv_output = (args + defaults)[:3]

    excel_df = well_data.read_sections(xls_file, sheetname)
    print(excel_df)
    well_data.csv_output(csv_output, excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])
