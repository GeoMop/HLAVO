import numpy as np
import pandas as pd

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


def get_measurements(time_step, measurements_struc, probe_data):
    """
    Retrieve measurement data for a given time step and structure.

    :param time_step: Time step index for which to extract data
    :param measurements_struc: Measurement structure defining locations and variables
    :param probe_data: List of measurement time series organized by depth
    :return: Dictionary mapping measurement names to lists of extracted values
    """
    pos_to_index = {-10: 0, -20: 1, -40: 2, -60: 3, -100: 4}

    measurements_dict = {}
    for measurement_name, measure_obj in measurements_struc.items():
        measurements = []
        for zp in measure_obj.z_pos:
            measurements.append(probe_data[pos_to_index[zp]].iloc[time_step])
        measurements_dict[measurement_name] = measurements

    return measurements_dict


def get_precipitations(df, time_step):
    """
    Compute distributed precipitation (inflow) over fixed time windows.

    :param df: DataFrame containing 'Inflow' and timestamp index
    :param time_step: Model time step used to compute total precipitation intervals
    :return: List of tuples (duration_minutes, inflow_value)
    """
    time_delta = 30  # minutes
    distribution_window = pd.Timedelta(minutes=time_delta)
    surface_area = 700

    df["DistributedInflow"] = 0.0
    df["Inflow_cm"] = -df["Inflow"] * 1000 / surface_area

    for date_time_round, row in df[df["Inflow_cm"].notna()].iterrows():
        start_time = date_time_round
        end_time = start_time + distribution_window
        inflow_value = row["Inflow_cm"]
        per_minute_value = inflow_value / time_delta

        mask = (df.index >= start_time) & (df.index < end_time)
        df.loc[mask, "DistributedInflow"] = per_minute_value

    # Group identical inflow values to determine total durations
    csv_timestep = 5  # minutes between data points
    df["value_change"] = df["DistributedInflow"] != df["DistributedInflow"].shift()
    df["group"] = df["value_change"].cumsum()

    df_grouped = (
        df.groupby("group")
        .agg(
            inflow_value=("DistributedInflow", "first"),
            count=("DistributedInflow", "count")
        )
        .reset_index(drop=True)
    )

    df_grouped["total_minutes"] = df_grouped["count"] * csv_timestep
    time_precipitation_list = list(zip(df_grouped["total_minutes"], df_grouped["inflow_value"]))

    return time_precipitation_list


def get_data_at_time_step(data, time_step):
    """
    Extract rows from a dataset at a fixed resampling interval.

    :param data: Input DataFrame with 'DateTime' column
    :param time_step: Sampling interval in minutes
    :return: Resampled DataFrame containing one record per time_step
    """
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.set_index('DateTime')
    data.index = data.index.round('1min')

    data_at_time_steps = data.resample(f'{int(time_step)}min', origin=data.index[0]).first().reset_index()
    return data_at_time_steps


def load_pr2_data(data):
    """
    Load and calibrate PR2 probe moisture data using predefined coefficients.

    :param data: Input DataFrame containing PR2 moisture readings
    :return: List of calibrated moisture arrays (per depth)
    """
    SoilMoistMin_0 = data["SoilMoistMin_0_smooth"]
    SoilMoistMin_1 = data["SoilMoistMin_1_smooth"]
    SoilMoistMin_2 = data["SoilMoistMin_2_smooth"]
    SoilMoistMin_3 = data["SoilMoistMin_3_smooth"]
    SoilMoistMin_4 = data["SoilMoistMin_4_smooth"]

    moistures = [SoilMoistMin_0, SoilMoistMin_1, SoilMoistMin_2, SoilMoistMin_3, SoilMoistMin_4]

    calibrated_moistures = []
    for moisture, pr2_coeff in zip(moistures, pr2_a3.values()):
        pr2_coeff = np.mean(pr2_coeff)
        calib_moisture = moisture / pr2_coeff * 0.3
        calibrated_moistures.append(calib_moisture)

    return calibrated_moistures


def load_odyssey_data(data):
    """
    Load odyssey sensor saturation data from a DataFrame.

    :param data: Input DataFrame containing odyssey readings
    :return: List of saturation arrays (per depth)
    """
    Saturation_oddysey_0 = data["odyssey_0"]
    Saturation_oddysey_1 = data["odyssey_1"]
    Saturation_oddysey_2 = data["odyssey_2"]
    Saturation_oddysey_3 = data["odyssey_3"]
    Saturation_oddysey_4 = data["odyssey_4"]

    saturations = [
        Saturation_oddysey_0,
        Saturation_oddysey_1,
        Saturation_oddysey_2,
        Saturation_oddysey_3,
        Saturation_oddysey_4
    ]

    return saturations


def preprocess_data(df):
    """
    Preprocess time series data for alignment and continuity:
      - Round timestamps to the nearest minute
      - Remove duplicates
      - Identify missing or irregular time gaps

    :param df: Raw input DataFrame containing a 'DateTime' column
    :return: Cleaned DataFrame indexed by rounded timestamps
    """
    df["DateTimeRound"] = pd.to_datetime(df["DateTime"]).dt.round("min")

    duplicates = df[df.duplicated(subset="DateTimeRound", keep=False)]
    duplicates = duplicates.sort_values("DateTimeRound")
    print("duplicates ", duplicates)

    df = df.drop_duplicates(subset="DateTimeRound", keep="first")
    df = df.sort_values("DateTimeRound").set_index("DateTimeRound")

    time_diffs = df.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]

    print("Gaps larger than 5 minutes:")
    print(large_gaps)
    print("len(df) ", len(df))

    return df


def load_data(train_measurements_struc, test_measurements_struc, data_csv, measurements_config):
    """
    Load and prepare sensor measurement data for training and testing.

    :param train_measurements_struc: Measurement structure for training data
    :param test_measurements_struc: Measurement structure for test data
    :param data_csv: Path to CSV file containing raw measurement data
    :param measurements_config: Configuration dictionary with probe type, model step, etc.
    :return: Tuple (train_measurements, test_measurements, precipitations, measurement_state_flag)
    """
    data = pd.read_csv(data_csv)
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
