#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
from hlavo.ingress.moist_profile import data_filter
# from hlavo.ingress import well_data   # for csv_output()


def main(args):
    final_df = data_filter.read_data()
    print(final_df)
    # well_data.csv_output("moist_out.csv", final_df)

    data_filter.apply_filter(final_df, 1, 1, "plot.png")

if __name__ == "__main__":
    main(sys.argv[1:])
