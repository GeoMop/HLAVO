import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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

    # substract atmospheric pressure
    teros_ids = ['A', 'B', 'C']
    for tid in teros_ids:
        teros31_merged[f'Pressure_{tid}{tid}'] = teros31_merged[f'Pressure_{tid}'] - teros31_merged['Pressure'] / 1000

    teros31_merged.set_index('DateTime', inplace=True)
    return teros31_merged

def read_atmospheric_data(base_dir):
    filename_pattern = os.path.join(base_dir, '**', 'atmospheric', '*.csv')
    data = read_data(filename_pattern)
    return data


def read_flow_data(base_dir):
    filename_pattern = os.path.join(base_dir, '**', 'flow', '*.csv')
    data = read_data(filename_pattern)
    return data


# def read_odyssey_data(filter=False):
#     pattern = os.path.join(base_dir, 'odyssey_*.csv')
#     data = read_data(pattern, dt_column='Date/Time', sep=',')
#     for i in range(5):
#         selected_column = f"sensor-{i+1} %"
#         new_col_name = f"odyssey_{i}"
#         data[selected_column] = data[selected_column]/100
#         data.rename(columns={selected_column: new_col_name}, inplace=True)
#     # FILTERING
#     if filter:
#         for i in range(0, 6):
#             selected_column = f"odyssey_{i}"
#             # pr2_a0_data_filtered = pr2_a0_data_filtered[(pr2_a0_data_filtered[selected_column] != 0)]
#             # Filter rows where a selected column is between 0 and 1
#             data = data[(data[selected_column] > 0.01) & (data[selected_column] <= 100)]
#
#     return data

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

    df.to_csv(os.path.join(output_folder, 'atm_data_filtered.csv'), index=True)

# Plot some columns using matplotlib
def plot_pr2_data(ax, df, title):
    cl_name = 'SoilMoistMin'
    columns = [f"{cl_name}_{i}" for i in range(6)]
    col_labels = [f"PR2 - {i} cm" for i in [10, 20, 30, 40, 60, 100]]

    filtered_df = df.dropna(subset=columns)

    for column, clb in zip(columns, col_labels):
        ax.plot(filtered_df.index, filtered_df[column], label=clb,
                marker='o', linestyle='-', markersize=2)
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

    filtered_df.to_csv(os.path.join(output_folder, 'pr2_data_filtered.csv'), index=True)

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

    col_labels = [f"Teros31 {i} - {j} cm" for i,j in zip(['A', 'B', 'C'],['10', '30', '100'])]

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

    filtered_df.to_csv(os.path.join(output_folder, 'teros31_data_filtered.csv'), index=True)


# Plot some columns using matplotlib
def plot_height_data(ax, df, title):
    plot_columns(ax, df, ['Height'], ylabel="Water height [mm]", startofdays=False)
    ax.set_title(title)

def plot_odyssey(ax, df):
    # columns = [f"odyssey_{i}" for i in range(5)]
    # col_labels = [f"Odyssey - {i} cm" for i in [10, 20, 40, 60, 100]]
    columns = [f"odyssey_{i}" for i in range(4)]
    col_labels = [f"Odyssey - {i} cm" for i in [10, 20, 40, 60]]
    for column, clb in zip(columns, col_labels):
        ax.plot(df.index, df[column], label=clb)

    add_start_of_days(df, ax)
    set_date_time_axis(ax)
    ax.set_title("Odyssey - Soil Moisture Mineral")
    ax.set_ylabel('Soil Moisture $\mathregular{[m^3\cdot m^{-3}]}$')
    ax.legend()

if __name__ == '__main__':
    # Define the directory structure
    hlavo_data_dir = '../../hlavo_data'
    base_dir = os.path.join(hlavo_data_dir, 'data_lab/data_lab_04')
    # Define the output folder
    output_folder = create_output_dir(os.path.join(hlavo_data_dir, 'OUTPUT', "lab_results_04"))

    odyssey = True
    # odyssey = False

    setup_plt_fontsizes()

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
    time_interval = {'start_date': '2024-12-13T12:00:00', 'end_date': '2025-02-04T23:59:59'}

    flow_data = select_time_interval(read_flow_data(base_dir), **time_interval)
    atm_data = select_time_interval(read_atmospheric_data(base_dir), **time_interval)
    pr2_data = select_time_interval(read_pr2_data(base_dir), **time_interval)

    teros31_data = read_teros31_data(base_dir)
    teros31_merged = merge_teros31_data(teros31_data, atm_data)
    teros31_merged = select_time_interval(teros31_merged, **time_interval)

    # Resample the data to get samples at every 5 minutes
    # meteo_data_resampled = meteo_data.resample('5min').first()
    # flow_data_resampled = flow_data.resample('1min').mean()
    # flow_data_resampled = flow_data

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_height_data(ax, flow_data, 'Water Height Over Time')
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'height_data.pdf'), format='pdf')

    flow_data_resampled = flow_data.resample('10min').mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_columns(ax, flow_data_resampled, ['Flux'])
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    ax.set_ylim([-0.1,0.05])
    ax.set_title('Water Flux Over Time')
    fig.savefig(os.path.join(output_folder, 'flux_data.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_atm_data(ax, atm_data, "Atmospheric data")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 1), bbox_transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'atm_data.pdf'), format='pdf')

    # plot_columns(ax, pr2_a0_data_filtered, ['SoilMoistMin_0', 'SoilMoistMin_5'], 'Soil Moisture Mineral')
    # plot_columns(ax, all_data, ['SoilMoistMin_0', 'SoilMoistMin_5'], 'Soil Moisture Mineral')
    # plot_pr2_data(ax, data_merged, "Rain vs Soil Moisture", **merging_dates)

    if odyssey:
        odyssey_data = read_odyssey_data(base_dir, filter=False, ids=[5])[0]
        # skip all data outside [0,1]
        # columns_to_check = [f"odyssey_{i}" for i in range(5)]  # List of columns to check
        columns_to_check = [f"odyssey_{i}" for i in range(4)]  # List of columns to check
        condition = (odyssey_data[columns_to_check] >= 0).all(axis=1) & (odyssey_data[columns_to_check] <= 1).all(axis=1)
        odyssey_data = odyssey_data.loc[condition]
        # odyssey_data = select_time_interval(odyssey_data[0], start_date='2024-10-01T00:00:00', end_date='2024-10-31T23:00:00')
        odyssey_data = select_time_interval(odyssey_data, **time_interval)



    data_merged = pd.merge(atm_data, pr2_data, how='outer', left_index=True, right_index=True, sort=True)

    # merging_dates = {'start_date': '2024-07-05', 'end_date': '2024-07-15'}

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pr2_data(ax, pr2_data, "Humidity vs Soil Moisture")
    ax.set_title('PR2 - Soil Moisture Mineral')
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'pr2_data.pdf'), format='pdf')

    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, teros31_merged, "Teros 31", diff=False)
    ax.set_title('Teros31 - Total Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'teros31_data_abs.pdf'), format='pdf')

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_teros31_data(ax, teros31_merged, "Teros 31", diff=True)
    ax.set_title('Teros31 - Matric Potential')
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'teros31_data_diff.pdf'), format='pdf')

    if odyssey:
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot_columns(ax, odyssey_data, columns=[f"odyssey_{i}" for i in range(4)], ylabel="", startofdays=True)
        plot_odyssey(ax, odyssey_data)
        fig.savefig(os.path.join(output_folder, "odyssey_data.pdf"), format='pdf')
        odyssey_data.to_csv(os.path.join(output_folder, 'odyssey_data_filtered.csv'), index=True)

    # plt.show()
