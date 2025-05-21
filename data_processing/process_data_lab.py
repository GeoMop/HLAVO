import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    return data

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
def plot_height_data(ax, df, title):
    plot_columns(ax, df, ['Height'], ylabel="Water height [mm]", startofdays=False)
    ax.set_title(title)

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
    plot_height_data(ax, flow_data, 'Water Height Over Time', cfg["output_dir"])
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'height_data.pdf'), format='pdf')


    flow_data_resampled = flow_data.resample('10min').mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_columns(ax, flow_data_resampled, ['Flux'])
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    ax.set_ylim([-0.1, 0.05])
    ax.set_title('Water Flux Over Time')
    fig.savefig(os.path.join(output_dir, 'flux_data.pdf'), format='pdf')


def read_all_data(cfg):
    base_dir = cfg["base_dir"]
    atm_data = read_atmospheric_data(base_dir)
    pr2_data = read_pr2_data(base_dir)
    teros31_data = read_teros31_data(base_dir)
    odyssey_id = cfg["odyssey_id"]
    ods_data = read_odyssey_data(base_dir, filter=False, ids=[odyssey_id])[0]

    # SHIFT UTC time to CEST (in Lab)
    # ods_data["DateTime"] = ods_data["DateTime"] + pd.to_timedelta(2, unit="h")
    ods_data.index = ods_data.index + pd.to_timedelta(2, unit="h")

    return [atm_data, pr2_data, *teros31_data, ods_data]


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


def main():
    setup_plt_fontsizes()
    cfg = select_inputs()

    process_flow_data(cfg)

    # all_dfs = [atm_data, pr2_data, *teros31_data, odyssey_data]
    all_dfs = read_all_data(cfg)
    merged_all = merge_all_dfs(all_dfs)
    data = select_time_interval(merged_all, **cfg["time_interval"])
    # data.to_parquet("hlavo_lab_merged_data_2025_03-05.parquet")
    data.to_csv(os.path.join(cfg["output_dir"], "hlavo_lab_merged_data.csv"))

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_atm_data(ax, data, "Atmospheric data")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 1), bbox_transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'atm_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_odyssey(ax, data)
    plot_pr2_data(ax, data, "PR2 vs Odyssey")
    legend = ax.get_legend()
    legend.set_bbox_to_anchor((1, 1))  # Move legend to top-right outside plot
    legend.set_loc('upper left')  # Anchor inside the box
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'pr2_vs_odyssey.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pr2_data(ax, data, "Humidity vs Soil Moisture")
    ax.set_title('PR2 - Soil Moisture Mineral')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'pr2_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, data, "Teros 31", diff=False)
    ax.set_title('Teros31 - Total Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'teros31_data_abs.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, data, "Teros 31", diff=True)
    ax.set_title('Teros31 - Matric Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], 'teros31_data_diff.pdf'), format='pdf')

    if cfg["odyssey"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot_columns(ax, odyssey_data, columns=[f"odyssey_{i}" for i in range(4)], ylabel="", startofdays=True)
        plot_odyssey(ax, data)
        fig.savefig(os.path.join(cfg["output_dir"], "odyssey_data.pdf"), format='pdf')
        # data.to_csv(os.path.join(cfg["output_dir"], 'odyssey_data_filtered.csv'), index=True)


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

    folder_id = "08"
    cfg = {
        "base_dir": os.path.join(hlavo_data_dir, f'data_lab/data_lab_{folder_id}'),
        "output_dir": create_output_dir(os.path.join(hlavo_data_dir, 'OUTPUT', f"lab_results_{folder_id}")),

        # full saturation experiment 2
        # "time_interval": {'start_date': '2025-03-26T11:00:00', 'end_date': '2025-05-15T12:00:00'},
        "time_interval": {'start_date': '2025-03-26T12:00:00', 'end_date': '2025-03-28T12:00:00'},

        # full saturation experiment 3
        # "time_interval": {'start_date': '2025-05-16T11:40:00', 'end_date': '2025-05-21T09:00:00'},

        "odyssey": True,
        "odyssey_id": 5,
        # "odyssey_id": 36
    }

    return cfg


if __name__ == '__main__':
    main()
    # plt.show()
