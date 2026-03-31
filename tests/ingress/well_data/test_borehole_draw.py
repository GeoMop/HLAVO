#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
from hlavo.ingress import well_data


def test_borehole_draw():
    xls_file, sheetname, csv_output = ["../hlavo/ingress/well_data/25_09_27_Odbery_Uhelna.xlsx", "List1", "borehole_water_draw_out.csv"]

    excel_df = well_data.read_draw(xls_file, sheetname)
    print(excel_df)
    # well_data.csv_output(csv_file=csv_output, df=excel_df)

if __name__ == "__main__":
    test_borehole_draw()
