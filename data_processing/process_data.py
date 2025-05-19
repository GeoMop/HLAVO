import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def setup_plt_fontsizes():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def select_time_interval(df, start_date=None, end_date=None):
    # Select data within the datetime interval
    if start_date is not None and end_date is not None:
        subset_df = df.loc[start_date:end_date]
    elif start_date is not None:
        subset_df = df.loc[start_date:]
    elif end_date is not None:
        subset_df = df.loc[:end_date]
    else:
        subset_df = df
    return subset_df

def add_start_of_days(df, ax):
    # Calculate total time span of the data
    time_span = df.index.max() - df.index.min()
    if time_span <= pd.Timedelta(days=6):
        # Generate timestamps every 4 hours
        if time_span <= pd.Timedelta(days=3):
            time_ticks = df.resample('2H').mean().index
        else:
            time_ticks = df.resample('4H').mean().index
        # Plot vertical lines
        for t in time_ticks:
            ax.axvline(t, color='lightgrey', linestyle='--', linewidth=0.5)

    # Add vertical lines at the start of each day
    start_of_days = df.resample('D').mean().index
    for day in start_of_days:
        ax.axvline(day, color='grey', linestyle='--', linewidth=0.5)

def set_date_time_axis(ax):
    ax.set_xlabel('DateTime')
    # Rotate the x-axis labels by 60 degrees
    for label in ax.get_xticklabels():
        label.set_rotation(30)

def parse_datetime_column(column):
    """
    Parses a datetime column with varying formats into a standardized datetime object.

    Args:
        column (pd.Series): A Pandas Series containing datetime strings or timestamps.

    Returns:
        pd.Series: A Series with standardized datetime objects.
    """
    def detect_and_parse(value):
        try:
            # Format 1: "25-09-2024 08:20:00"
            return datetime.strptime(value, '%d-%m-%Y %H:%M:%S')
        except ValueError:
            pass

        try:
            # Format 2: "2024-09-25T16:50:24" (ISO format)
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        try:
            # Format 3: Unix timestamp in milliseconds
            timestamp = int(value)
            return pd.to_datetime(timestamp, unit='ms')
        except (ValueError, OverflowError):
            pass

        # If all parsing attempts fail, return NaT
        return pd.NaT

    # Apply detection and parsing to the entire column
    return column.apply(detect_and_parse)


def read_data(file_pattern, dt_column='DateTime', sep=';'):
    """
    Reads data from CSV file with file structure given by :param:`file_pattern`
    """
    # Use glob to get all CSV files in the directory structure
    csv_files = glob.glob(file_pattern, recursive=True)

    # Initialize an empty list to store dataframes
    dataframes = []

    # Read each CSV file and append to the list
    for file in csv_files:
        df = pd.read_csv(file, sep=sep)
        dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Convert DateTime to datetime and other columns to float
    # Using errors='coerce' to convert invalid datetime entries to NaT
    # Get the first non-null value to determine the format

    # Parse the datetime column
    merged_df[dt_column] = parse_datetime_column(merged_df[dt_column])

    # Drop rows with NaT in 'DateTime' column
    merged_df = merged_df.dropna(subset=[dt_column])

    float_columns = merged_df.columns.drop(dt_column)
    # when converting to floats, make NaN where invalid strings are present
    merged_df[float_columns] = merged_df[float_columns].apply(pd.to_numeric, errors='coerce')
    # Drop rows with NaN values in float columns if necessary
    # merged_df = merged_df.dropna(subset=float_columns)

    # Sort the DataFrame by DateTime
    merged_df = merged_df.sort_values(by=dt_column)
    merged_df.set_index(dt_column, inplace=True)
    return merged_df


def read_pr2_data(base_dir, filter=False):
    pr2_all_data = []
    # for each PR2 sensor
    for a in range(0, 2):
        pr2_pattern = os.path.join(base_dir, '**', 'pr2_sensor_' + str(a), '*.csv')
        data = read_data(pr2_pattern)
        # FILTERING
        if filter:
            for i in range(0, 6):
                selected_column = 'SoilMoistMin_' + str(i)
                # pr2_a0_data_filtered = pr2_a0_data_filtered[(pr2_a0_data_filtered[selected_column] != 0)]
                # Filter rows where a selected column is between 0 and 1
                data = data[(data[selected_column] > 0.01) & (data[selected_column] <= 1)]

        pr2_all_data.append(data)
    return pr2_all_data


def read_odyssey_data(base_dir, filter=False, ids=[]):
    ods_data = []
    # for each Odyssey sensor
    for a in ids:
        pattern = os.path.join(base_dir, 'data_odyssey', '*U' + str(a).zfill(2) + '*.csv')
        data = read_data(pattern, dt_column='dateTime', sep=',')
        # data.rename(columns={'dateTime': 'DateTime'}, inplace=True)
        # data.set_index('DateTime', inplace=True)
        data.index.name = 'DateTime'
        for i in range(5):
            # moisture
            selected_column = f"s{i+1}"
            new_col_name = f"odyssey_{i}"
            data[selected_column] = data[selected_column]/100
            data.rename(columns={selected_column: new_col_name}, inplace=True)
            # temperature
            data.rename(columns={f"s{i+1}t": f"temp_{i}"}, inplace=True)
        # ambient temperature
        data.rename(columns={f"s11t": f"ambient_temp"}, inplace=True)

        # FILTERING
        if filter:
            for i in range(0, 6):
                selected_column = f"odyssey_{i}"
                # pr2_a0_data_filtered = pr2_a0_data_filtered[(pr2_a0_data_filtered[selected_column] != 0)]
                # Filter rows where a selected column is between 0 and 1
                data = data[(data[selected_column] > 0.01) & (data[selected_column] <= 100)]

        ods_data.append(data)
    return ods_data

# Plot some columns using matplotlib
def plot_columns(ax, df, columns, ylabel='Values', startofdays=True):
    for column in columns:
        # dat = df[(df[column] > 0.01) & (df[column] < 1)]
        # ax.plot(dat.index, dat[column], label=column,
        #         marker='o', linestyle='-', markersize=2)
        ax.plot(df.index, df[column], label=column)

    # plt.show()
    # Add vertical lines at the start of each day
    if startofdays:
        add_start_of_days(df, ax)

    set_date_time_axis(ax)
    ax.set_ylabel(ylabel)
    ax.legend()

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

# Plot some columns using matplotlib
def plot_moisture_rain(ax, df, title, start_date=None, end_date=None):
    cl_name = 'SoilMoistMin'
    columns = [f"{cl_name}_{i}" for i in range(6)]

    # Select data within the datetime interval
    if start_date is not None and end_date is not None:
        interval_df = df.loc[start_date:end_date]
    elif start_date is not None:
        interval_df = df.loc[start_date:]
    elif end_date is not None:
        interval_df = df.loc[:end_date]
    else:
        interval_df = df
    filtered_df = interval_df.dropna(subset=columns)

    for column in columns:
        # ax.plot(filtered_df.index, filtered_df[column], label=column,
        #         marker='o', linestyle='-', markersize=2)
        dat = filtered_df[(filtered_df[column] > 0.01) & (filtered_df[column] < 1)]
        window_size = 5  # You can adjust the window size
        smoothed_dat = dat.rolling(window=window_size).max()
        ax.plot(smoothed_dat.index, smoothed_dat[column], label=column,
                marker='o', linestyle='-', markersize=2)

    add_start_of_days(df, ax)

    ax.set_xlabel('DateTime')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.legend()

    cl_name = 'RainGauge'
    rain_df = interval_df[interval_df[cl_name]>0].dropna(subset=cl_name)
    ax2 = ax.twinx()
    ax2.plot(rain_df.index, rain_df[cl_name], 'r', label='Rain',
             marker='o', linestyle='-', markersize=2)
    ax2.set_ylabel('Rain [ml/min]')
    ax2.legend(loc='center right')


def plot_moisture_rain_comparison(ax, df, title, start_date=None, end_date=None):
    # PR2
    cl_name = 'SoilMoistMin'
    depths = [0,2]
    columns = [f"{cl_name}_{i}" for i in depths]

    # Select data within the datetime interval
    if start_date is not None and end_date is not None:
        interval_df = df.loc[start_date:end_date]
    elif start_date is not None:
        interval_df = df.loc[start_date:]
    elif end_date is not None:
        interval_df = df.loc[:end_date]
    else:
        interval_df = df
    filtered_df = interval_df.dropna(subset=columns)

    for column in columns:
        # ax.plot(filtered_df.index, filtered_df[column], label=column,
        #         marker='o', linestyle='-', markersize=2)
        dat = filtered_df[(filtered_df[column] > 0.01) & (filtered_df[column] < 1)]
        window_size = 5  # You can adjust the window size
        smoothed_dat = dat.rolling(window=window_size).max()
        ax.plot(smoothed_dat.index, smoothed_dat[column], label=column,
                marker='o', linestyle='-', markersize=2)

    # Odyssey
    columns = [f"odyssey_{i}" for i in depths]
    for column in columns:
        ax.plot(interval_df.index, interval_df[column], label=column, marker='o', linestyle='-', markersize=2)

    add_start_of_days(df, ax)

    ax.set_xlabel('DateTime')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.legend()

    cl_name = 'RainGauge'
    rain_df = interval_df[interval_df[cl_name]>0].dropna(subset=cl_name)
    ax2 = ax.twinx()
    ax2.plot(rain_df.index, rain_df[cl_name], 'r', label='Rain',
             marker='o', linestyle='-', markersize=2)
    ax2.set_ylabel('Rain [ml/min]')
    ax2.legend(loc='upper right')

def create_output_dir(path):
    # Check if the folder exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)

if __name__ == '__main__':
    setup_plt_fontsizes()
    # # Define the directory structure
    # base_dir = '..'
    # meteo_pattern = os.path.join(base_dir, '**', 'meteo', '*.csv')
    # # Define the output folder
    # output_folder = create_output_dir("meteo_station_01")

    # Define the directory structure
    hlavo_data_dir = '../../hlavo_data'
    base_dir = os.path.join(hlavo_data_dir, 'data_station')
    meteo_pattern = os.path.join(base_dir, '**', 'meteo', '*.csv')
    # Define the output folder
    output_folder = create_output_dir(os.path.join(hlavo_data_dir, 'OUTPUT', "meteo_station_01"))

    odyssey = False

    # time_interval = {'start_date': '2024-06-28T10:00:00', 'end_date': '2024-09-10T23:59:59'}
    time_interval = {'start_date': '2024-12-01T00:00:00', 'end_date': '2025-02-04T23:59:59'}


    meteo_data = select_time_interval(read_data(meteo_pattern), **time_interval)
    # Resample the data to get samples at every 15 minutes
    # meteo_data_resampled = meteo_data.resample('15min').first()
    meteo_data_resampled = meteo_data.resample('15min').mean()
    # meteo_data_resampled = meteo_data

    # Example: Plot WindSpeed and Temperature_Mean
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_columns(ax, meteo_data_resampled, ['Humidity_Mean', 'Temperature_Mean', 'RainGauge'])
    ax.set_title('Humidity, Temperature, RainGauge Over Time')
    # plot_columns(ax, merged_df, ['Humidity_Mean', 'Temperature_Mean'], 'Humidity and Temperature Over Time')
    fig.savefig(os.path.join(output_folder, 'meteo_data.pdf'), format='pdf')

    # PR2 - a0 - s Oddyssey U01 u meteo stanice
    # PR2 - a1 - s Oddyssey U04 pod stromy
    if odyssey:
        # time_interval = {'start_date': '2024-06-01T00:00:00', 'end_date': '2024-06-20T23:59:59'}
        time_interval = {'start_date': '2024-06-01T00:00:00', 'end_date': '2024-11-04T23:59:59'}
        odyssey_data = read_odyssey_data(base_dir, filter=False, ids=[1, 2, 3, 4])
        odyssey_names = [f"U0{i+1}" for i in range(4)]
        for i in [0,1,2,3]:
            # skip all data outside [0,1]
            odyssey_columns = [f"odyssey_{i}" for i in range(5)]  # List of columns to check
            condition = (odyssey_data[i][odyssey_columns] >= 0).all(axis=1) & (odyssey_data[i][odyssey_columns] <= 1).all(axis=1)
            odyssey_data[i] = odyssey_data[i].loc[condition]
            odyssey_data[i] = select_time_interval(odyssey_data[i], **time_interval)

            fig, ax = plt.subplots(figsize=(10, 6))
            plot_columns(ax, odyssey_data[i], columns=odyssey_columns)
            ax.set_title(f"Odyssey {odyssey_names[i]} - Soil Moisture Mineral")
            fig.tight_layout()
            fig.savefig(os.path.join(output_folder, f"odyssey_data_{odyssey_names[i]}.pdf"), format='pdf')

    # exit(0)
    pr2_data = read_pr2_data(base_dir, filter=False)
    pr2_names = [f"a{i}" for i in range(2)]
    # merging_dates = {'start_date': '2024-07-05', 'end_date': '2024-07-15'}
    merging_dates = time_interval

    for i in range(len(pr2_data)):
        pr2_data[i] = select_time_interval(pr2_data[i], **time_interval)

        # Merge the meteo and pr2 dataframes on DateTime using outer join
        pr2_data_merged = pd.merge(meteo_data, pr2_data[i], how='outer', left_index=True, right_index=True, sort=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot_columns(ax, pr2_a0_data_filtered, ['SoilMoistMin_0', 'SoilMoistMin_5'], 'Soil Moisture Mineral')
        # plot_columns(ax, all_data, ['SoilMoistMin_0', 'SoilMoistMin_5'], 'Soil Moisture Mineral')
        # plot_moisture_rain(ax, pr2_data_merged, "Rain vs Soil Moisture", **merging_dates)
        plot_pr2_data(ax, pr2_data[i], f"PR2 {pr2_names[i]} - Soil Moisture Mineral")
        # ax.set_title(f"PR2 {pr2_names[i]} - Soil Moisture Mineral")
        fig.savefig(os.path.join(output_folder, f"pr2_data_{pr2_names[i]}.pdf"), format='pdf')
        # filtered_df.to_csv(os.path.join(output_folder, 'f"pr2_data_{pr2_names[i]}_filtered.csv"), index=True)
        # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_columns(ax, pr2_data[0], ['SoilMoistMin_0', 'SoilMoistMin_5'])
    # plot_columns(ax, pr2_data[1], ['SoilMoistMin_0', 'SoilMoistMin_5'])
    # ax.set_title('Soil Moisture Mineral')
    # # plot_columns(ax, all_data, ['SoilMoistMin_0', 'SoilMoistMin_5'], 'Soil Moisture Mineral')
    # # plot_moisture_rain(ax, all_data, "Rain vs Soil Moisture", start_date='2024-07-05', end_date='2024-07-15')
    # # fig.savefig('pr2_data.pdf', format='pdf')
    # plt.show()


    # PR2 - a0 - s Oddyssey U01 u meteo stanice
    # PR2 - a1 - s Oddyssey U04 pod stromy
    # Merge PR2 and Odyssey dataframes on DateTime using outer join
    # if odyssey:
    #     U01_data_merged = pd.merge(pr2_data_merged_a0, odyssey_data[0], how='outer', left_index=True, right_index=True, sort=True)
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     plot_moisture_rain_comparison(ax, U01_data_merged, "Rain vs Soil Moisture", **merging_dates)
    #     ax.set_title('Soil Moisture Mineral')
    #     fig.savefig('U01_data_merged.pdf', format='pdf')
    #
    #     U02_data_merged = pd.merge(pr2_data_merged_a1, odyssey_data[1], how='outer', left_index=True, right_index=True, sort=True)
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     plot_moisture_rain_comparison(ax, U02_data_merged, "Rain vs Soil Moisture", **merging_dates)
    #     ax.set_title('Soil Moisture Mineral')
    #     fig.savefig('U02_data_merged.pdf', format='pdf')
