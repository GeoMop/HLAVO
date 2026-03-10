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



# def load_zarr_data():
#     zarr_file_path = "/HLAVO/tests/ingress/moist_profile/test_storage"
#
#     if os.path.exists(zarr_file_path):
#         print("location exists")
#     zarr_file = zarr.open(zarr_file_path, mode='r+')


def load_zarr_data(train_measurements_struc, test_measurements_struc, zarr_dir, scheme_file, measurements_config):
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


    min_idx = 0
    max_idx = len(data) - 1
    measurements_time_step = measurements_config["model_time_step"] * measurements_config["model_n_time_steps_per_iter"]

    data = preprocess_data(data[min_idx:max_idx])

    timestamps = data["DateTime"].to_numpy()

    measurement_state_flag = []
    if "State" in data:
        measurement_state_flag = data["State"].to_numpy()

    probe = measurements_config.get("probe")
    if probe == "pr2":
        probe_data = load_pr2_data(data)
    elif probe == "odyssey":
        probe_data = load_odyssey_data(data)
    else:
        raise AttributeError(f"Probe '{probe}' not supported. Use 'pr2' or 'odyssey'.")

    precipitations = get_precipitations(data, time_step=measurements_time_step)

    train_measurements = []
    test_measurements = []

    for i in range(len(data)):
        measurement_train = get_measurements(i, train_measurements_struc, probe_data)
        measurement_test = get_measurements(i, test_measurements_struc, probe_data)

        train_measurements.append(train_measurements_struc.encode(measurement_train))
        test_measurements.append(test_measurements_struc.encode(measurement_test))

    return train_measurements, test_measurements, precipitations, measurement_state_flag, timestamps
