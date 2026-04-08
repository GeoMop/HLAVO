#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

import sys
from hlavo.ingress.moist_profile import data_filter


def test_data_filter():
    site_ids = [1, 2, 3]
    depth_levels = [1, 2, 3, 4, 5]

    final_df = data_filter.read_data(site_ids, depth_levels)
    print(final_df)
    #data_filter.apply_filter(final_df, site_ids, depth_levels, "plot_all")

if __name__ == "__main__":
    test_data_filter()
