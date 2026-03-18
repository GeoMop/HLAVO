#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
from hlavo.ingress import well_data


def main(args):
    defaults = ["../hlavo/ingress/well_data/25_09_27_Odbery_Uhelna.xlsx", "List1", "borehole_water_draw_out.csv"]
    xls_file, sheetname, csv_output = (args + defaults)[:3]

    excel_df = well_data.read_draw(xls_file, sheetname)
    print(excel_df)
    # well_data.csv_output(csv_file=csv_output, df=excel_df)

if __name__ == "__main__":
    main(sys.argv[1:])
