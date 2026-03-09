#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
sys.path.append('../../../hlavo/deep_model/vrty')
import conversion_data


def main(argv):
    if (len(argv) != 3):
        argv = ["../../../hlavo/deep_model/vrty/Vrty_souradnice_perforace.xlsx", "List1", "Vrty_uprav.csv"]
        # temporary hack, set args automatically if they are not set
        #print("Invalid number of input args, must be 3! ")
        #sys.exit(1)

    excel_df = conversion_data.read_sections(section_file=argv[0], sheetname=argv[1])
    print(excel_df)
    # conversion_data.csv_output("borehole_water_section_out.csv", excel_df)

if __name__ == "__main__":
   main(sys.argv[1:])
