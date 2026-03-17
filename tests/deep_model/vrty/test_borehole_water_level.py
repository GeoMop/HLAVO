#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka



import sys
sys.path.append('../../../hlavo/deep_model/vrty')
import well_data

def main():
    final_df = well_data.read_water_level()
    print(final_df)
    #well_data.csv_output("borehole_water_level_out.csv", final_df)

if __name__ == "__main__":
   main()
