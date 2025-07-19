import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#from sklearn.linear_model import LinearRegression

pr2_a3 = {'10': [0.2418, 0.2772, 0.3306, 0.3547, 0.3449],
          '20': [0.2417, 0.2768, 0.3304, 0.3552, 0.3436],
          '40': [0.2421, 0.2772, 0.3303, 0.3541, 0.3444],
          '60': [0.2423, 0.2769, 0.3303, 0.3542, 0.3434],
          '100': [0.3302, 0.3549, 0.342]
          }

pr2_a1 = {'10': [0.2437, 0.2761, 0.3244, 0.3495, 0.328],
          '20': [0.2445, 0.2763, 0.3243, 0.3498, 0.3277],
          '40': [0.2429, 0.2762, 0.3242, 0.3492, 0.3283],
          '60': [0.2431, 0.2759, 0.3242, 0.3494, 0.3283],
          '100': [0.2438, 0.3248]
          }


def get_measurements(time_step, measurements_struc, probe_data):
    pos_to_index = {-10: 0, -20: 1, -40: 2, -60: 3, -100: 4}

    measurements_dict = {}
    for measurement_name, measure_obj in measurements_struc.items():
        measurements = []
        for zp in measure_obj.z_pos:
            # if measurement_name == "moisture":
            #     measurement_data = moistures
            # elif measurement_name == "saturation":
            #     measurement_data = saturations
            measurements.append(probe_data[pos_to_index[zp]][time_step])

        measurements_dict[measurement_name] = measurements

    return measurements_dict


# def get_missing_timesteps(df):
#     # Ensure DateTime is datetime and sorted
#     df["DateTime"] = pd.to_datetime(df["DateTime"])
#     df = df.sort_values("DateTime").reset_index(drop=True)
#
#     # Step 1: Infer expected time step (in seconds or minutes)
#     dt_diffs = df["DateTime"].diff().dropna()
#     most_common_step = dt_diffs.mode()[0]  # Use mode as the most frequent step
#
#     # Step 2: Create a full datetime range with that step
#     full_range = pd.date_range(start=df["DateTime"].min(),
#                                end=df["DateTime"].max(),
#                                freq=most_common_step)
#
#     # Step 3: Find missing timestamps
#     actual_timestamps = df["DateTime"]
#     missing_timestamps = full_range.difference(actual_timestamps)
#
#     # Output
#     print("Expected time step:", most_common_step)
#     print("Number of missing timestamps:", len(missing_timestamps))
#     print("Missing timestamps:")
#     print(missing_timestamps)


def get_precipitations(df, time_step):
    # Define distribution window (e.g., 30 minutes)
    time_delta = 30
    distribution_window = pd.Timedelta(minutes=time_delta)
    surface_area = 700

    # Initialize column for distributed inflow
    df["DistributedInflow"] = 0.0

    df["Inflow_cm"] = -df["Inflow"]*1000/surface_area

    print(df["Inflow"])
    print(df["Inflow_cm"])

    # Loop through each non-NaN inflow row
    for date_time_round, row in df[df["Inflow_cm"].notna()].iterrows():
        print("date_time_round ", date_time_round)
        print("row ", row)
        start_time = date_time_round #row["DateTimeRound"]
        end_time = start_time + distribution_window
        inflow_value = row["Inflow_cm"]

        # Uniform value per minute
        per_minute_value = inflow_value / time_delta

        # Mask for the 30-minute window starting at this row
        mask = (df.index >= start_time) & (df.index < end_time)
        #mask = (df["DateTimeRound"] >= start_time) & (df["DateTimeRound"] < end_time)

        # Assign constant value to all rows within that time window
        df.loc[mask, "DistributedInflow"] = per_minute_value

    #get_missing_timesteps(df)

    print(df[:10])
    print("len(df) ", len(df))

    ###
    # Calculate precipitation list

    # To ignore rounds etc., consider fixed timestep of measurements
    csv_timestep = 5
    # Ensure sorted by time
    #df["DateTimeRound"] = pd.to_datetime(df["DateTimeRound"])
    #df = df.sort_values("DateTimeRound").reset_index(drop=True)

    # Identify changes in DistributedInflow to group consecutive identical values
    df["value_change"] = df["DistributedInflow"] != df["DistributedInflow"].shift()
    df["group"] = df["value_change"].cumsum()

    # Count rows in each group (each row = 5 minutes)
    df_grouped = (
        df.groupby("group")
        .agg(
            inflow_value=("DistributedInflow", "first"),
            count=("DistributedInflow", "count")
        )
        .reset_index(drop=True)
    )

    # Multiply row count by 5 minutes
    df_grouped["total_minutes"] = df_grouped["count"] * csv_timestep

    # Final result
    time_precipitation_list = list(zip(df_grouped["total_minutes"], df_grouped["inflow_value"]))

    return time_precipitation_list


def get_data_at_time_step(data, time_step):
    # Step 1: Parse datetime
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    print("original data ", data)
    # Step 2: Set datetime as index
    data = data.set_index('DateTime')
    # Round timestamps to the nearest minute to get clean alignment
    data.index = data.index.round('1min')
    print(data[:20])
    # Resample every 30 minutes — pick one row per time_step-min interval
    data_at_time_steps = data.resample('{}min'.format(int(time_step)), origin=data.index[0]).first().reset_index()

    print(data_at_time_steps[:10])

    return data_at_time_steps


def load_pr2_data(data, measurements_time_step, window_size=50):
    #data = data[min_idx:max_idx]

    data_at_time_steps = data #get_data_at_time_step(data, measurements_time_step)

    print("data_at_time_steps ", data_at_time_steps)

    SoilMoistMin_0 = data_at_time_steps["SoilMoistMin_0"]
    SoilMoistMin_1 = data_at_time_steps["SoilMoistMin_1"]
    SoilMoistMin_2 = data_at_time_steps["SoilMoistMin_2"]
    SoilMoistMin_3 = data_at_time_steps["SoilMoistMin_3"]
    SoilMoistMin_4 = data_at_time_steps["SoilMoistMin_4"]

    SoilMoistMin_0_smooth = SoilMoistMin_0.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_1_smooth = SoilMoistMin_1.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_2_smooth = SoilMoistMin_2.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_3_smooth = SoilMoistMin_3.rolling(window=window_size, min_periods=1).mean()
    SoilMoistMin_4_smooth = SoilMoistMin_4.rolling(window=window_size, min_periods=1).mean()

    print("SoilMoistMin_0_smooth ", SoilMoistMin_0_smooth[:15])

    moistures = [SoilMoistMin_0_smooth, SoilMoistMin_1_smooth, SoilMoistMin_2_smooth, SoilMoistMin_3_smooth,
                 SoilMoistMin_4_smooth]

    return [SoilMoistMin_0, SoilMoistMin_1, SoilMoistMin_2, SoilMoistMin_3, SoilMoistMin_4]

    #return moistures

    calibrated_moistures = []
    for moisture, pr2_coeff in zip(moistures, pr2_a3.values()):
        pr2_coeff = np.mean(pr2_coeff)
        moisture = moisture / pr2_coeff * 30

        calibrated_moistures.append(moisture)

    return calibrated_moistures


def load_odyssey_data(data, window_size=50):
    Saturation_oddysey_0 = data["odyssey_0"]
    Saturation_oddysey_1 = data["odyssey_1"]
    Saturation_oddysey_2 = data["odyssey_2"]
    Saturation_oddysey_3 = data["odyssey_3"]
    Saturation_oddysey_4 = data["odyssey_4"]

    Saturation_oddysey_0_smooth = Saturation_oddysey_0.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_1_smooth = Saturation_oddysey_1.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_2_smooth = Saturation_oddysey_2.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_3_smooth = Saturation_oddysey_3.rolling(window=window_size, min_periods=1).mean()
    Saturation_oddysey_4_smooth = Saturation_oddysey_4.rolling(window=window_size, min_periods=1).mean()

    saturations = [Saturation_oddysey_0_smooth, Saturation_oddysey_1_smooth, Saturation_oddysey_2_smooth,
                   Saturation_oddysey_3_smooth, Saturation_oddysey_4_smooth]

    return saturations


def preprocess_data(df):
    # Step 1: Round DateTime to nearest full minute
    df["DateTimeRound"] = pd.to_datetime(df["DateTime"]).dt.round("min")

    # Step 2: Find duplicated timestamps after rounding
    duplicates = df[df.duplicated(subset="DateTimeRound", keep=False)]
    # Step 3: Sort for easy viewing
    duplicates = duplicates.sort_values("DateTimeRound")
    # Display duplicates
    print("duplicates ", duplicates)

    # Step 2: Drop duplicates, keep first
    df = df.drop_duplicates(subset="DateTimeRound", keep="first")

    # Step 3: Sort and set as index
    df = df.sort_values("DateTimeRound").set_index("DateTimeRound")

    # Compute time differences between consecutive entries
    time_diffs = df.index.to_series().diff()

    # Find where the gap is greater than 5 minutes
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]

    # Show the gaps
    print("Gaps larger than 5 minutes:")
    print(large_gaps)

    print("len(df) ", len(df))

    # # Step 3: Sort and set as index
    # df = df.sort_values("DateTimeRound").set_index("DateTimeRound")
    #
    # # Step 3: Create full 5-minute range
    # full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    # df_full = df.reindex(full_index)
    # df_full.index.name = "DateTimeRound"
    #
    # # List columns to exclude from filling
    # exclude_cols = ["DateTime", "DateTimeRound", "loggerUid", "Inflow"]
    #
    # # Select columns to fill (all except excluded)
    # cols_to_fill = [col for col in df_full.columns if col not in exclude_cols]
    # for col in cols_to_fill:
    #     # Ensure float dtype
    #     df_full[col] = df_full[col].astype("float64")
    #     # Compute rolling average
    #     rolling_avg = df_full[col].rolling(window=3, center=True, min_periods=1).mean()
    #     # Fill missing values only
    #     df_full[col] = df_full[col].where(df_full[col].notna(), rolling_avg)

    return df


def load_data(train_measurements_struc, test_measurements_struc, data_csv,  measurements_config):
    #data_csv = "/home/martin/Documents/HLAVO/kolona/hlavo_lab_merged_data_2025_06_12.csv"
    data = pd.read_csv(data_csv)
    min_idx = 0
    max_idx = 4500  # 7209
    mean_window_size = 50

    measurements_time_step = measurements_config["model_time_step"] * measurements_config["model_n_time_steps_per_iter"]

    data = preprocess_data(data[min_idx:max_idx])

    probe = measurements_config["probe"] if "probe" in measurements_config else None
    if probe == "pr2":
        probe_data = load_pr2_data(data, measurements_time_step, window_size=mean_window_size)
    elif probe == "odyssey":
        probe_data = load_odyssey_data(data, measurements_time_step, window_size=mean_window_size)
    else:
        raise AttributeError("Probe: {} nor supported. Use 'pr2' or  'odyssey'".format(probe))

    precipitations = get_precipitations(data, time_step=measurements_time_step)

    train_measurements = []
    test_measurements = []
    for i in range(len(data)):
        measurement_train = get_measurements(i, train_measurements_struc, probe_data)
        measurement_test = get_measurements(i, test_measurements_struc, probe_data)

        train_measurements.append(train_measurements_struc.encode(measurement_train))
        test_measurements.append(test_measurements_struc.encode(measurement_test))

    return train_measurements, test_measurements, precipitations
