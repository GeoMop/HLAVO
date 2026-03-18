import os
import zarr_fuse as zf
import pandas as pd
from pathlib import Path

pr2_a3 = {
    '10': [0.2418, 0.2772, 0.3306, 0.3547, 0.3449],
    '20': [0.2417, 0.2768, 0.3304, 0.3552, 0.3436],
    '40': [0.2421, 0.2772, 0.3303, 0.3541, 0.3444],
    '60': [0.2423, 0.2769, 0.3303, 0.3542, 0.3434],
    '100': [0.3302, 0.3549, 0.342]
}

pr2_a1 = {
    '10': [0.2437, 0.2761, 0.3244, 0.3495, 0.328],
    '20': [0.2445, 0.2763, 0.3243, 0.3498, 0.3277],
    '40': [0.2429, 0.2762, 0.3242, 0.3492, 0.3283],
    '60': [0.2431, 0.2759, 0.3242, 0.3494, 0.3283],
    '100': [0.2438, 0.3248]
}


def load_measurments_data(scheme_file):
    """
    Load and prepare sensor measurement data for training and testing.

    :param train_measurements_struc: Measurement structure for training data
    :param test_measurements_struc: Measurement structure for test data
    :param data_csv: Path to CSV file containing raw measurement data
    :param measurements_config: Configuration dictionary with probe type, model step, etc.
    :return: Tuple (train_measurements, test_measurements, precipitations, measurement_state_flag)
    """
    ## load bukov ##
    #
    import yaml
    # structure_file = yaml.safe_load('bukov.zarr/zarr.json')
    # print("structure file ", structure_file)

    #options = {"STORE_URL": "to_be_overwritten_by_schema", "WORKDIR": "to_be_overwritten_by_schema"}
    scheme_file_path = Path(scheme_file)
    print(scheme_file_path)
    #schema = zf.schema.deserialize(scheme_file_path)
    #schema.ds.ATTRS['WORKDIR'] = zarr_dir
    #zf.remove_store(schema, **options)
    root = zf.open_store(scheme_file_path)#, **options)

    print(root)

    print(root["Uhelna"].dataset)

    profiles = root["Uhelna"]["profiles"]

    ds = profiles

    print(ds.dataset)


    return ds.dataset


def load_meteo_data(scheme_file):
    scheme_file_path = Path(scheme_file)
    root = zf.open_store(scheme_file_path)

    print("meteo root ", root)
    meteo_data = root["chmi_aladin_10m"]


    print("meteo data ", meteo_data)
