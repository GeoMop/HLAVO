import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#from sklearn.linear_model import LinearRegression


# def load_data(data_dir, n_samples):
#
#     #data_dir = "/home/martin/Documents/HLAVO/soil_model/24_10_01_full_saturation/"
#
#     pr2_data_file = os.path.join(data_dir, "pr2_data_filtered.csv")
#     odyssey_data_file = os.path.join(data_dir, "odyssey_data_filtered.csv")
#
#     # Load the CSV file into a pandas DataFrame
#     pr2_data = pd.read_csv(pr2_data_file)
#     odyssey_data = pd.read_csv(odyssey_data_file)
#
#     min_idx = 264
#     max_idx = 1992
#
#     x_range = np.arange(min_idx, max_idx)
#
#     # Define the window size (e.g., 3-point moving average)
#     window_size = 50
#
#     odyssey_0 = odyssey_data["odyssey_0"][min_idx:max_idx]
#     odyssey_1 = odyssey_data["odyssey_1"][min_idx:max_idx]
#     odyssey_2 = odyssey_data["odyssey_2"][min_idx:max_idx]
#     odyssey_3 = odyssey_data["odyssey_3"][min_idx:max_idx]
#     odyssey_4 = odyssey_data["odyssey_4"][min_idx:max_idx]
#
#     pr2_0 = pr2_data["SoilMoistMin_0"][min_idx:max_idx]
#     pr2_1 = pr2_data["SoilMoistMin_1"][min_idx:max_idx]
#     pr2_2 = pr2_data["SoilMoistMin_2"][min_idx:max_idx]
#     pr2_3 = pr2_data["SoilMoistMin_3"][min_idx:max_idx]
#     pr2_4 = pr2_data["SoilMoistMin_4"][min_idx:max_idx]
#     pr2_5 = pr2_data["SoilMoistMin_5"][min_idx:max_idx]
#
#     # Apply the moving average filter
#     pr2_0_smoothed = pr2_0.rolling(window=window_size, min_periods=1).mean()
#     pr2_1_smoothed = pr2_1.rolling(window=window_size, min_periods=1).mean()
#     pr2_2_smoothed = pr2_2.rolling(window=window_size, min_periods=1).mean()
#     pr2_5_smoothed = pr2_5.rolling(window=window_size, min_periods=1).mean()
#     pr2_4_smoothed = pr2_4.rolling(window=window_size, min_periods=1).mean()
#     pr2_3_smoothed = pr2_3.rolling(window=window_size, min_periods=1).mean()
#
#     # One value per hour
#     pr2_0_smoothed = pr2_0_smoothed[::12]
#     pr2_1_smoothed = pr2_1_smoothed[::12]
#     pr2_2_smoothed = pr2_2_smoothed[::12]
#     pr2_3_smoothed = pr2_3_smoothed[::12]
#     pr2_4_smoothed = pr2_4_smoothed[::12]
#     pr2_5_smoothed = pr2_5_smoothed[::12]
#
#     #train_data = np.array([pr2_0_smoothed, pr2_1_smoothed, pr2_2_smoothed, pr2_4_smoothed, pr2_5_smoothed]).T
#     train_data = np.array([pr2_5_smoothed[:n_samples]]).T
#     test_data = np.array([pr2_3_smoothed[:n_samples]]).T
#
#     return train_data, test_data

def get_measurements(time_step, measurements_struc, moistures, saturations):
    pos_to_index = {-10: 0, -20: 1, -40: 2, -60: 3, -100: 4}

    measurements_dict = {}
    for measurement_name, measure_obj in measurements_struc.items():
        measurements = []
        for zp in measure_obj.z_pos:
            if measurement_name == "moisture":
                measurement_data = moistures
            elif measurement_name == "saturation":
                measurement_data = saturations
            measurements.append(measurement_data[pos_to_index[zp]][time_step])

        measurements_dict[measurement_name] = measurements

    return measurements_dict

def get_precipitations(df):
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Define distribution window (e.g., 30 minutes)
    distribution_window = pd.Timedelta(minutes=30)

    # Initialize column for distributed inflow
    df["DistributedInflow"] = 0.0

    # Loop through each non-NaN Inflow entry
    for i, row in df[df["Inflow"].notna()].iterrows():
        start_time = row["DateTime"]
        end_time = start_time + distribution_window
        inflow_value = row["Inflow"]

        # Mask for time window
        mask = (df["DateTime"] >= start_time) & (df["DateTime"] < end_time)
        time_steps = df[mask]

        if not time_steps.empty:
            per_step = inflow_value / len(time_steps)
            df.loc[mask, "DistributedInflow"] += per_step

    return df["DistributedInflow"]


def load_data(train_measurements_struc, test_measurements_struc, data_csv):
    #data_csv = "/home/martin/Documents/HLAVO/kolona/hlavo_lab_merged_data_2025_06_12.csv"
    data = pd.read_csv(data_csv)


    min_idx = 0
    max_idx = 4500  # 7209

    SoilMoistMin_0 = data["SoilMoistMin_0"][min_idx:max_idx]
    SoilMoistMin_1 = data["SoilMoistMin_1"][min_idx:max_idx]
    SoilMoistMin_2 = data["SoilMoistMin_2"][min_idx:max_idx]
    SoilMoistMin_3 = data["SoilMoistMin_3"][min_idx:max_idx]
    SoilMoistMin_4 = data["SoilMoistMin_4"][min_idx:max_idx]

    window_size = 50
    SoilMoistMin_0_smooth = SoilMoistMin_0.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_1_smooth = SoilMoistMin_1.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_2_smooth = SoilMoistMin_2.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_3_smooth = SoilMoistMin_3.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_4_smooth = SoilMoistMin_4.rolling(window=window_size, min_periods=1).mean()

    moistures = [SoilMoistMin_0_smooth, SoilMoistMin_1_smooth, SoilMoistMin_2_smooth, SoilMoistMin_3_smooth, SoilMoistMin_4_smooth]

    Saturation_oddysey_0 = data["odyssey_0"][min_idx:max_idx]
    Saturation_oddysey_1 = data["odyssey_1"][min_idx:max_idx]
    Saturation_oddysey_2 = data["odyssey_2"][min_idx:max_idx]
    Saturation_oddysey_3 = data["odyssey_3"][min_idx:max_idx]
    Saturation_oddysey_4 = data["odyssey_4"][min_idx:max_idx]

    Saturation_oddysey_0_smooth = Saturation_oddysey_0.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_1_smooth = Saturation_oddysey_1.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_2_smooth = Saturation_oddysey_2.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_3_smooth = Saturation_oddysey_3.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_4_smooth = Saturation_oddysey_4.rolling(window=window_size, min_periods=1).mean()

    saturations = [Saturation_oddysey_0_smooth, Saturation_oddysey_1_smooth, Saturation_oddysey_2_smooth, Saturation_oddysey_3_smooth, Saturation_oddysey_4_smooth]

    precipitations = get_precipitations(data)
    precipitations = precipitations[min_idx:max_idx]

    train_measurements = []
    test_measurements = []
    for i in range(min_idx, max_idx):
        measurement_train = get_measurements(i, train_measurements_struc, moistures, saturations)
        measurement_test = get_measurements(i, test_measurements_struc, moistures, saturations)

        # print("measurement_train ", measurement_train)
        # print("train_measurements_struc.encode(measurement_train) ", train_measurements_struc.encode(measurement_train))
        #
        # print("measurement_test ", measurement_test)
        # print("test_measurements_struc.encode(measurement_test) ", test_measurements_struc.encode(measurement_test))

        train_measurements.append(train_measurements_struc.encode(measurement_train))
        test_measurements.append(test_measurements_struc.encode(measurement_test))

    # print("train_measurements ", train_measurements)
    # print("test measurements ", test_measurements)
    #
    # print("precipitations ", precipitations)

    return train_measurements, test_measurements, precipitations



