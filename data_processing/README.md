# HLAVO Data aquisition

- `dataflow_grab.py` - automatically process webpage of DataFlow (xpert.nz) and downloads data reports from Oddysey Xtreem

- `process_data.py` - reads various data from meteo station (meteo CSV, PR2, Oddysey), filter, cut and plot data
- `process_data_lab.py` - reads various data from laboratory (atm, flow, PR2, Oddysey), filter, cut and plot data
- supposes data in dir structure:
    - hlavo_data
        - data_lab
        - data_station