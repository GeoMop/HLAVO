import os
import glob

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


from process_data import create_output_dir, read_data, read_odyssey_data, setup_plt_fontsizes, select_time_interval, \
    plot_columns, add_start_of_days, set_date_time_axis

def read_pr2_data(base_dir, filter=False):
    # for each PR2 sensor
    filename_pattern = os.path.join(base_dir, '**', 'pr2_sensor', '*.csv')
    data = read_data(filename_pattern)
    # FILTERING
    if filter:
        for i in range(0, 6):
            selected_column = 'SoilMoistMin_' + str(i)
            # pr2_a0_data_filtered = pr2_a0_data_filtered[(pr2_a0_data_filtered[selected_column] != 0)]
            # Filter rows where a selected column is between 0 and 1
            data = data[(data[selected_column] > 0.01) & (data[selected_column] <= 1)]

    # remove unnecessary columns
    keep_columns = [c for c in data.columns if not any(b in c for b in ["Voltage", "rawADC"])]
    return data[keep_columns]

def read_teros31_data(base_dir, filter=False):
    data = []
    sensor_names = ['A', 'B', 'C']
    for a in range(0, 3):
        filename_pattern = os.path.join(base_dir, '**', 'teros31_sensor_' + str(a), '*.csv')
        data_chunk = read_data(filename_pattern)
        # print(data_chunk)
        data_chunk.rename(columns={'Pressure': f"Pressure_{sensor_names[a]}"}, inplace=True)
        data_chunk.rename(columns={'Temperature': f"Temperature_{sensor_names[a]}"}, inplace=True)
        # print(data_chunk)
        # FILTERING
        # if filter:
            # for i in range(0, 6):
            #     selected_column = 'SoilMoistMin_' + str(i)
            #     # pr2_a0_data_filtered = pr2_a0_data_filtered[(pr2_a0_data_filtered[selected_column] != 0)]
            #     # Filter rows where a selected column is between 0 and 1
            #     data_chunk = data_chunk[(data_chunk[selected_column] > 0.01) & (data_chunk[selected_column] <= 1)]

        data.append(data_chunk)
    return data

def merge_teros31_data(teros31_data, atm_data):
    teros31_merged = teros31_data[0]
    for i in range(len(teros31_data)-1):
        teros31_merged = pd.merge(teros31_merged, teros31_data[i+1], how='outer', left_index=True, right_index=True,
                                  sort=True)

    # add atmospheric data for pressure
    teros31_merged = pd.merge_asof(teros31_merged, atm_data, on='DateTime', direction='nearest')

    teros31_merged.set_index('DateTime', inplace=True)
    return teros31_merged


def teros31_pressure_diff(df):
    # substract atmospheric pressure
    teros_ids = ['A', 'B', 'C']
    for tid in teros_ids:
        df[f'Pressure_{tid}{tid}'] = df[f'Pressure_{tid}'] - df['Pressure'] / 1000


def read_atmospheric_data(base_dir):
    filename_pattern = os.path.join(base_dir, '**', 'atmospheric', '*.csv')
    data = read_data(filename_pattern)
    return data


def read_flow_data(base_dir):
    filename_pattern = os.path.join(base_dir, '**', 'flow', '*.csv')
    data = read_data(filename_pattern)
    return data

def read_inflow_data(base_dir):
    filename_pattern = os.path.join(base_dir, 'inflow.csv')
    data = read_data(filename_pattern)
    return data

def add_inflow_times(df, ax):
    # Filter to only the rows where 'Inflow' is not NaN (i.e., matched sparse points)
    sparse_points = df[df['Inflow'].notna()]

    # Add blue dotted vertical lines and optional labels
    for timestamp, row in sparse_points.iterrows():
        inflow_val = row['Inflow']
        ax.axvline(x=timestamp, color='blue', linestyle='dotted', linewidth=1)

        # Optional: annotate inflow value at the top of the line
        ymax = ax.get_ylim()[1]
        ax.text(timestamp, ymax, f"{inflow_val:.2f}", color='blue',
                ha='center', va='bottom', fontsize=8, rotation=90)

# Plot some columns using matplotlib
def plot_atm_data(ax, df, title):
    column = "Pressure"
    ax.plot(df.index, df[column]/1000, label=column, marker='o', linestyle='-', markersize=2, color='red')
    ax.set_ylabel('Pressure [kPa]')

    ax2 = ax.twinx()
    column = "Temperature"
    ax2.plot(df.index, df[column], label=column, marker='o', linestyle='-', markersize=2, color='blue')
    ax2.set_ylabel('Temperature $\mathregular{[^oC]}$')
    # ax2.legend()
    # ax2.legend(loc='center top')

    add_start_of_days(df, ax)
    set_date_time_axis(ax)

    ax.set_title(title)
    # ax.legend()

# Plot some columns using matplotlib
def plot_pr2_data(ax, df, title):
    cl_name = 'SoilMoistMin'
    columns = [f"{cl_name}_{i}" for i in range(6)]
    col_labels = [f"PR2 - {i} cm" for i in [10, 20, 30, 40, 60, 100]]

    filtered_df = df.dropna(subset=columns)

    for column, clb in zip(columns, col_labels):
        ax.plot(filtered_df.index, filtered_df[column], label=clb, #marker='o',
                linestyle='-', markersize=2)
        # dat = filtered_df[(filtered_df[column] > 0.01) & (filtered_df[column] < 1)]
        # window_size = 5  # You can adjust the window size
        # smoothed_dat = dat.rolling(window=window_size).max()
        # ax.plot(smoothed_dat.index, smoothed_dat[column], label=column,
        #         marker='o', linestyle='-', markersize=2)

    add_start_of_days(df, ax)
    set_date_time_axis(ax)

    ax.set_ylabel('Soil Moisture $\mathregular{[m^3\cdot m^{-3}]}$')
    ax.set_title(title)
    ax.legend()

    # cl_name = 'Humidity'
    # hum_df = interval_df[interval_df[cl_name]>0].dropna(subset=cl_name)
    # ax2 = ax.twinx()
    # ax2.plot(hum_df.index, hum_df[cl_name], 'r', label='Humidity',
    #          marker='o', linestyle='-', markersize=2)
    # ax2.set_ylabel('Humidity [%]')
    # ax2.legend(loc='center right')


# Plot some columns using matplotlib
def plot_teros31_data(ax, df, title, diff=True):
    cl_name = 'Pressure'
    if diff:
        columns = [f"{cl_name}_{i}" for i in ['AA', 'BB' ,'CC']]
    else:
        columns = [f"{cl_name}_{i}" for i in ['A', 'B', 'C']]

    col_labels = [f"Teros31 {i} - {j} cm" for i,j in zip(['A', 'B', 'C'],['10', '40', '100'])]

    # filtered_df = interval_df.dropna(subset=columns)
    filtered_df = df[(df != 0).all(axis=1)]

    for column, clb in zip(columns, col_labels):
        ax.plot(filtered_df.index, filtered_df[column], label=clb,
                marker='o', markersize=2)
        # get last line color
        color = ax.get_lines()[-1].get_color()

        interpolated_df = filtered_df.interpolate()
        ax.plot(filtered_df.index, interpolated_df[column], label='_nolegend_', linestyle='-', color=color)
        # ax.plot(filtered_df.index, filtered_df[column], label=column,
        #         marker='o', linestyle='-', markersize=2)
        # dat = filtered_df[(filtered_df[column] > 0.01) & (filtered_df[column] < 1)]
        # window_size = 5  # You can adjust the window size
        # smoothed_dat = dat.rolling(window=window_size).max()
        # ax.plot(smoothed_dat.index, smoothed_dat[column], label=column,
        #         marker='o', linestyle='-', markersize=2)

    add_start_of_days(df, ax)
    set_date_time_axis(ax)

    ax.set_ylabel('Potential [kPa]')
    ax.set_title(title)
    ax.legend()

    # filtered_df.to_csv(os.path.join(output_folder, 'teros31_data_filtered.csv'), index=True)


# Plot some columns using matplotlib
def plot_height_data(ax, df, title, output_dir):
    ax.set_title(title)

    select_df = df.dropna(subset=['Height']).copy()
    # select_df = df[["Height"]].copy()
    h_y = select_df["Height"].values
    h_x = (select_df.index - select_df.index[0]).total_seconds()
    # Add column with time index
    select_df.loc[:, "TimeIndex"] = h_x

    # Apply a rolling median filter for smoothing
    # window_size = 50  # Adjust window size based on data characteristics
    # select_df["Smoothed_Height"] = median_filter(select_df["Height"], size=window_size)

    # start_height = select_df["Smoothed_Height"].iloc[:100].mean()
    # select_df["Smoothed_Height"] = select_df["Smoothed_Height"] - start_height

    # Detect jumps by finding significant negative changes in height
    # threshold = select_df["Smoothed_Height"].diff().quantile(0.01)  # 5th percentile as jump threshold
    # jumps = select_df["Smoothed_Height"].diff() < threshold

    # Compute cumulative correction to maintain continuity
    # cumulative_correction = np.cumsum(jumps * -select_df["Smoothed_Height"].diff())
    # Apply correction to the height column
    # select_df["Cumulative_Height"] = select_df["Smoothed_Height"] + cumulative_correction

    # 1. Find jumps in measured data
    # threshold = -100
    threshold = -0.75 * (np.max(h_y) - np.min(h_y))
    print(f"jump threshold: {threshold}")
    jumps = select_df["Height"].diff() < threshold

    correction = np.zeros(len(select_df))
    jump_end_indices = np.where(jumps)[0]
    print(f"N jumps: {len(jump_end_indices)}")
    jump_start_indices = []

    if len(jump_end_indices) == 0:
        select_df.loc[:, 'Height_Smooth'] = savgol_filter(select_df["Height"], window_length=51, polyorder=3, delta=3)
        select_df["Height_Cumulative"] = select_df['Height_Smooth']
    else:

        # 2. Smooth measured data between jumps
        # Apply Savitzky-Golay filter
        # window_length must be odd and >= polyorder + 2
        h_smooth = select_df["Height"].values.copy()
        pairs = ([(0, jump_end_indices[0])]
                 + [(jump_end_indices[i], jump_end_indices[i + 1]) for i in range(len(jump_end_indices) - 1)]
                 + [(jump_end_indices[-1], len(h_x))] )
        for sid, eid in pairs:
            ids = np.arange(sid, eid)
            h_smooth[ids] = savgol_filter(h_smooth[ids], window_length=51, polyorder=3, delta=3)
        select_df.loc[:, 'Height_Smooth'] = h_smooth

        # 3. Iterate between jumps and connect the data using tangent approximation
        data_col = "Height_Smooth"
        h_y = select_df[data_col].values

        for end_idx in jump_end_indices:
            start_idx = end_idx-1
            jump_start_indices.append(start_idx)

            # Use polyfit to fit the data before and after jump
            # and compute tangent
            def get_slope(ia, ib, ider, label, plot=True):
                # from scipy.stats import linregress
                # slope, intercept, r_value, p_value, std_err = linregress(h_x[ia:ib], h_y[ia:ib])
                # print(r_value, p_value, std_err)
                # if plot:
                #     y_fit = slope * h_x[ia:ib] + intercept
                #     ax.plot(select_df.index[ia:ib], y_fit, label=label)
                ids = np.arange(ia, ib)
                coeffs = np.polyfit(h_x[ids], h_y[ids], deg=2)  # or deg=3
                poly = np.poly1d(coeffs)
                # First derivative (gives the slope)
                dpoly = np.polyder(poly)
                # Evaluate slope at the last x-value
                x_tangent = h_x[ids[ider]]
                slope = dpoly(x_tangent)
                if plot:
                    y_fit = slope * (h_x[ids] - x_tangent) + poly(x_tangent)
                    ax.plot(select_df.index[ids], y_fit, label=label+"_der")
                    y_fit = poly(h_x[ids])
                    ax.plot(select_df.index[ia:ib], y_fit, label=label)

                return slope

            # Average the two tangents
            segment_length = 300
            slope_a = get_slope(start_idx - segment_length, start_idx, -1,f"Tangent{end_idx}a", plot=False)
            slope_b = get_slope(end_idx, end_idx + segment_length, 0,f"Tangent{end_idx}b", plot=False)
            slope_avg = (slope_a + slope_b) / 2
            print(slope_a, slope_b, slope_avg)

            print(start_idx, end_idx)
            assert end_idx-start_idx == 1
            time_interval = select_df.index[end_idx] - select_df.index[start_idx]
            time_interval_s = time_interval.total_seconds()

            # Create correction the times of jumps
            correction[end_idx] = slope_avg * time_interval_s
            print(correction[start_idx-2:end_idx+2])

        # Compute cumulative correction to maintain continuity
        cumulative_correction = np.cumsum(jumps * -select_df[data_col].diff() + correction)
        # Apply correction to the height column
        select_df["Height_Cumulative"] = select_df[data_col] + cumulative_correction

    # 4. Interpolate by spline the cummulative height function
    print('create spline')
    from scipy.interpolate import UnivariateSpline
    x = select_df["TimeIndex"].values  # x-axis in seconds
    y = select_df['Height_Cumulative'].values
    w = np.isnan(y)
    y[w] = 0.
    # Fit a smoothing spline, Adjust 's' to control smoothness
    spline = UnivariateSpline(x, y, w=~w, s=len(w)/5, ext=2)
    # Evaluate smooth curve and its derivative
    print('evaluate spline')
    select_df['Height_Spline'] = spline(x)
    print("N spline knots: ", len(spline.get_knots()))
    select_df['Flux_Spline'] = - spline.derivative()(x)

    print('plot')
    # plot_columns(ax, select_df, ['Smoothed_Height', 'Cumulative_Height'],
    #              ylabel="Cumulative Water height [mm]", startofdays=False)

    # 'Smooth_Cumulative_Height', 'Deriv_Height'
    plot_columns(ax, select_df,
                 ['Height', 'Height_Smooth', 'Height_Cumulative', 'Height_Spline', 'Flux_Spline'],
                 ylabel="Cumulative Water height [mm]", startofdays=False)
    # plot_columns(ax, select_df, ['Height', 'Smoothed_Height', 'Cumulative_Height', 'Smooth_Cumulative_Height', 'Deriv_Height'],
    #                           ylabel="Cumulative Water height [mm]", startofdays=False)

    # Plot vertical lines
    for sid, eid in zip(jump_start_indices, jump_end_indices):
        ax.axvline(select_df.index[sid], color='lightgrey', linestyle=':', linewidth=0.5)
        ax.axvline(select_df.index[eid], color='lightgrey', linestyle='--', linewidth=0.5)

    select_df.to_csv(os.path.join(output_dir, 'height_filtered.csv'), index=True, sep=';')

    return select_df, spline

def plot_odyssey(ax, df):
    columns = [f"odyssey_{i}" for i in range(5)]
    col_labels = [f"Odyssey - {i} cm" for i in [10, 20, 40, 60, 100]]
    # columns = [f"odyssey_{i}" for i in range(4)]
    # col_labels = [f"Odyssey - {i} cm" for i in [10, 20, 40, 60]]
    for column, clb in zip(columns, col_labels):
        ax.plot(df.index, df[column], label=clb)

    add_start_of_days(df, ax)
    set_date_time_axis(ax)
    ax.set_title("Odyssey - Soil Moisture Mineral")
    ax.set_ylabel('Soil Moisture $\mathregular{[m^3\cdot m^{-3}]}$')
    ax.legend()


def process_flow_data(cfg):
    flow_data = select_time_interval(read_flow_data(cfg["base_dir"]), **cfg["time_interval"])

    fig, ax = plt.subplots(figsize=(10, 6))
    flow_data, h_spline = plot_height_data(ax, flow_data, 'Water Height Over Time', cfg["output_dir"])
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'height_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_columns(ax, flow_data, ['Flux_Spline'])
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    ax.set_title('Water Flux Over Time [spline]')
    fig.savefig(os.path.join(cfg["output_dir"], 'flux_data_spline.pdf'), format='pdf')

    flow_data_resampled = flow_data.resample('10min').mean()
    # flow_data_resampled = flow_data
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_columns(ax, flow_data_resampled, ['Flux', 'Flux_Spline'])
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    ax.set_ylim([-0.02, 0.0])
    ax.set_title('Water Flux Over Time')
    fig.savefig(os.path.join(cfg["output_dir"], 'flux_data.pdf'), format='pdf')

    # add PumpIn, PumpOut columns (for 9-11)
    # add RainHeight (for 12-)

    keep_columns = [c for c in flow_data.columns if c not in ["Height", "Flux", "TimeIndex", "Height_Smooth"]]
    flow_data = flow_data[keep_columns]
    return flow_data


def read_all_data(cfg):
    base_dir = cfg["base_dir"]
    inflow_data = read_inflow_data(base_dir)
    atm_data = read_atmospheric_data(base_dir)
    pr2_data = read_pr2_data(base_dir, filter=True)
    # pr2_data = read_pr2_data(base_dir, filter=False)
    teros31_data = read_teros31_data(base_dir)
    odyssey_id = cfg["odyssey_id"]

    if cfg["odyssey"]:
        ods_data = read_odyssey_data(base_dir, filter=False, ids=[odyssey_id])[0]

        # SHIFT UTC time to CEST (in Lab)
        # ods_data["DateTime"] = ods_data["DateTime"] + pd.to_timedelta(2, unit="h")
        ods_data.index = ods_data.index + pd.to_timedelta(2, unit="h")
    else:
        ods_data = None

    return [atm_data, pr2_data, *teros31_data, ods_data, inflow_data]


def merge_all_dfs(dfs):

    # Start from the first DataFrame
    merged_df = dfs[0]
    for df in dfs[1:]:
        if df is not None:
            merged_df = pd.merge_asof(merged_df, df,
                                      on='DateTime',
                                      tolerance=pd.Timedelta('3min'),
                                      direction='nearest')

    merged_df.set_index('DateTime', inplace=True)
    teros31_pressure_diff(merged_df)
    return merged_df


def merge_flow_data(dfs, flow_data):
    merged_df = pd.merge_asof(dfs, flow_data,
                              on='DateTime',
                              tolerance=pd.Timedelta('5s'),
                              direction='nearest')
    merged_df.set_index('DateTime', inplace=True)
    return merged_df

def merge_inflow_data(cfg, dfs, inflow_data):

    # Ensure sorted and integer indexed
    dfs = dfs.reset_index().sort_values('DateTime').reset_index(drop=True)
    inflow_data = inflow_data.reset_index().sort_values('DateTime').reset_index(drop=True)

    # Create a new column in fine data to hold the matched sparse values
    dfs['Inflow'] = np.nan
    # Define your max allowed time difference
    max_diff = pd.Timedelta('10min')
    # Track which fine timestamps have already been used
    used_indices = set()

    for i, row in inflow_data.iterrows():
        sparse_time = row['DateTime']
        inflow_value = row['Inflow']

        # Calculate absolute time difference to each fine timestamp
        time_diffs = (dfs['DateTime'] - sparse_time).abs()

        # Mask already used fine indices and apply tolerance
        mask = (time_diffs <= max_diff) & (~dfs.index.isin(used_indices))
        if mask.any():
            nearest_idx = time_diffs[mask].idxmin()
            dfs.at[nearest_idx, 'Inflow'] = inflow_value
            used_indices.add(nearest_idx)
        else:
            print(f"Warning: No match found for sparse timestamp {sparse_time}")

    # # check inflow merge - compute total inflow over time
    # # inflow_data_selected = select_time_interval(inflow_data, **cfg["time_interval"])
    # total_inflow = inflow_data["Inflow"].sum(numeric_only=True)
    # total_inflow_merged = dfs["Inflow"].sum(numeric_only=True)
    # assert total_inflow == total_inflow_merged
    # print('total_inflow: ', total_inflow)

    dfs.set_index('DateTime', inplace=True)
    return dfs


def main():
    setup_plt_fontsizes()
    cfg = select_inputs()

    flow_data = process_flow_data(cfg)
    # plt.show()
    # exit(0)

    # all_dfs = [atm_data, pr2_data, *teros31_data, odyssey_data]
    all_dfs = read_all_data(cfg)
    merged_all = merge_all_dfs(all_dfs[:-1])
    merged_all = merge_inflow_data(cfg, merged_all, all_dfs[-1])
    merged_all = merge_flow_data(merged_all, flow_data)

    data = select_time_interval(merged_all, **cfg["time_interval"])
    # data.to_parquet("hlavo_lab_merged_data_2025_03-05.parquet")
    data.to_csv(os.path.join(cfg["output_dir"], "hlavo_lab_merged_data.csv"))

    # check inflow merge - compute total inflow over time
    inflow_data = select_time_interval(all_dfs[-1], **cfg["time_interval"])
    total_inflow = inflow_data["Inflow"].sum(numeric_only=True)
    total_inflow_merged = data["Inflow"].sum(numeric_only=True)
    print('total_inflow: ', total_inflow)
    assert total_inflow == total_inflow_merged

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_atm_data(ax, data, "Atmospheric data")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 1), bbox_transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'atm_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pr2_data(ax, data, "Humidity vs Soil Moisture")
    add_inflow_times(data, ax)
    ax.set_title('PR2 - Soil Moisture Mineral')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'pr2_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, data, "Teros 31", diff=False)
    add_inflow_times(data, ax)
    ax.set_title('Teros31 - Total Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'teros31_data_abs.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, data, "Teros 31", diff=True)
    add_inflow_times(data, ax)
    ax.set_title('Teros31 - Matric Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'teros31_data_diff.pdf'), format='pdf')

    if cfg["odyssey"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot_columns(ax, odyssey_data, columns=[f"odyssey_{i}" for i in range(4)], ylabel="", startofdays=True)
        plot_odyssey(ax, data)
        add_inflow_times(data, ax)
        fig.savefig(os.path.join(cfg["output_dir"], "odyssey_data.pdf"), format='pdf')
        # data.to_csv(os.path.join(cfg["output_dir"], 'odyssey_data_filtered.csv'), index=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_odyssey(ax, data)
        plot_pr2_data(ax, data, "PR2 vs Odyssey")
        add_inflow_times(data, ax)
        legend = ax.get_legend()
        legend.set_bbox_to_anchor((1, 1))  # Move legend to top-right outside plot
        legend.set_loc('upper left')  # Anchor inside the box
        fig.tight_layout()
        fig.savefig(os.path.join(cfg["output_dir"], 'pr2_vs_odyssey.pdf'), format='pdf')


def select_inputs():
    # Define the directory structure
    hlavo_data_dir = '../../hlavo_data'

    # Odyssey is in UTC time
    # Lab is in CEST (Central European Summer Time) = UTC+2

    # odyssey
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2024-12-14T11:59:59'}
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2025-01-20T23:59:59'}

    # time_interval = {'start_date': '2024-12-13T18:00:00', 'end_date': '2024-12-13T20:30:00'}

    # height start
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2024-12-14T06:59:59'}

    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2024-12-27T23:59:59'}
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2024-12-31T23:59:59'}
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2025-01-20T23:59:59'}

    # full saturation experiment 1
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2025-02-04T23:59:59'}
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2024-12-20T23:59:59'}

    # full saturation experiment 2
    # time_interval = {'start_date': '2025-03-26T11:00:00', 'end_date': '2025-04-01T12:00:00'}
    # time_interval = {'start_date': '2025-03-26T12:00:00', 'end_date': '2025-03-28T12:00:00'}
    # time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2025-02-04T23:59:59'}
    # time_interval = {'start_date': '2025-03-12T12:30:00', 'end_date': '2025-03-26T11:00:00'}
    # time_interval = {'start_date': '2025-03-26T11:00:00', 'end_date': '2025-05-15T12:00:00'}
    # time_interval = {'start_date': '2025-05-10T11:00:00', 'end_date': '2025-05-15T12:00:00'}

    # full saturation experiment 3
    # time_interval = {'start_date': '2025-05-16T11:40:00', 'end_date': '2025-05-21T09:00:00'}

    # folder_id = "09"
    # folder_id = "11"
    folder_id = "09-11"
    # folder_id = "12"
    # folder_id = "13"
    # folder_id = "14"
    cfg = {
        "base_dir": os.path.join(hlavo_data_dir, f'data_lab/data_lab_{folder_id}'),
        "output_dir": create_output_dir(os.path.join(hlavo_data_dir, 'OUTPUT', f"lab_results_{folder_id}")),

        # full saturation experiment 2
        # "time_interval": {'start_date': '2025-03-26T12:00:00', 'end_date': '2025-05-15T12:00:00'},
        # "time_interval": {'start_date': '2025-03-26T12:00:00', 'end_date': '2025-03-28T12:00:00'},

        # full saturation experiment 3
        # "time_interval": {'start_date': '2025-05-16T11:35:00', 'end_date': '2025-06-13T09:00:00'},
        # "time_interval": {'start_date': '2025-06-02T08:00:00', 'end_date': '2025-06-03T09:00:00'},

        # full saturation - release experiment 3
        # "time_interval": {'start_date': '2025-06-13T00:00:00', 'end_date': '2025-06-23T10:00:00'},
        # "time_interval": {'start_date': '2025-06-13T00:00:00', 'end_date': '2025-06-25T10:00:00'},

        # folder_id = "11"
        # full saturation - release experiment 4
        # "time_interval": {'start_date': '2025-06-25T18:00:00', 'end_date': '2025-08-06T11:00:00'},
        # "time_interval": {'start_date': '2025-06-25T18:00:00', 'end_date': '2025-08-13T15:00:00'},

        # folder_id = "9-11"
        # full saturation - release experiment 3-4 merged
        "time_interval": {'start_date': '2025-05-16T0:00:00', 'end_date': '2025-08-15T15:00:00'},


        # folder_id = "12"
        # raining experiment - 8 rain regimes for 48 hours, accidentaly 24h watchdog
        # "time_interval": {'start_date': '2025-08-14T15:00:00', 'end_date': '2025-08-16T15:00:00'},
        # "time_interval": {'start_date': '2025-08-14T15:00:00', 'end_date': '2025-08-28T10:00:00'},

        # folder_id = "13"
        # raining experiment - 8 rain regimes for 48 hours
        # "time_interval": {'start_date': '2025-08-26T08:00:00', 'end_date': '2025-08-29T10:00:00'},

        # folder_id = "14"
        # raining experiment - 8 rain regimes for 48 hours
        # "time_interval": {'start_date': '2025-08-26T08:00:00', 'end_date': '2025-09-17T00:00:00'},


        "odyssey": True,
        # "odyssey_id": 5,
        "odyssey_id": 36
    }

    return cfg


if __name__ == '__main__':
    main()
    # plt.show()
