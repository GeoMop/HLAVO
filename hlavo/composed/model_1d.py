import numpy as np
from dask.distributed import Queue
from hlavo.kalman.kalman import KalmanFilter
from hlavo.ingress.moist_profile.load_zarr_data import load_measurments_data, load_meteo_data
from hlavo.composed.data_1d_to_3d import Data1DTo3D

# ---------------------------------------------------------------------------
# 1D model class
# ---------------------------------------------------------------------------

class Model1D:
    def __init__(self, site_id, initial_state=0.0, work_dir=None, model_kalman_config_dict=None, seed=None):
        """
        Initialize the 1D model instance for a specific site.

        :param int site_id: Identifier of the site used for selecting measurements.
        :param float initial_state: Initial state value for the model.
        :param Path work_dir: Working directory used by the Kalman filter.
        :param dict model_kalman_config_dict: Configuration dictionary for Kalman filter setup.
        :param int seed: Random seed for reproducibility.
        """

        self.site_id = site_id
        self.state = initial_state
        self.measurements_dataset = None

        # Initialize Kalman filter from configuration
        self.kalman = KalmanFilter.from_config(
            work_dir,
            config_dict=model_kalman_config_dict,
            verbose=False,
            seed=seed
        )

        # Load meteorological data required for model forcing
        self._get_meteo_data()

        # Prepare measurement dataset and initialize UKF
        self.ukf = self._prepare_kalman_measurements()

    def _get_meteo_data(self):
        """
        Load and preprocess meteorological dataset required by the model.
        :return: None
        """
        # Load meteorological dataset from configured scheme file
        meteo_dataset = load_meteo_data(self.kalman.measurements_config["meteo_scheme_file"])

        self.meteo_dataset = meteo_dataset.sel(site_id=self.site_id)

        # # TEMPORARY: Shift year to 2025 to align with measurement dataset
        # # TODO: Remove once datasets are temporally consistent
        # import pandas as pd
        #
        # orig_date_time = pd.to_datetime(self.meteo_dataset["date_time"].values)
        #
        # self.meteo_dataset = self.meteo_dataset.assign_coords(
        #     date_time=pd.to_datetime({
        #         "year": 2025,
        #         "month": orig_date_time.month,
        #         "day": orig_date_time.day,
        #         "hour": orig_date_time.hour,
        #         "minute": orig_date_time.minute,
        #         "second": orig_date_time.second,
        #     })
        # )

        print("self.meteo_dataset ", self.meteo_dataset)

    def _prepare_kalman_measurements(self):
        """
        Load measurement dataset, filter it for the current site, and initialize the Kalman filter.

        :return: Configured Unscented Kalman Filter (UKF) instance.
        """
        # Load measurement dataset from scheme file
        measurements_xarray = load_measurments_data(
            scheme_file=self.kalman.measurements_config["measurements_scheme_file"]
        )

        # Select data for the specific site
        self.measurements_dataset = measurements_xarray.sel(site_id=self.site_id)

        # TODO: Define proper measurement covariance matrix (R)
        kalman_R_matrix = Model1D.calculate_kalman_R_matrix(len(self.measurements_dataset["depth_level"]))

        # Initialize and return configured Kalman filter
        return self.kalman.set_kalman_filter(kalman_R_matrix)

    def get_dataset_values_for_time(self, dataset, start_time: np.datetime64, stop_time: np.datetime64):
        """
        Retrieve dataset part within a specified time window.

        :param numpy.datetime64 start_time: Inclusive start of the time window.
        :param numpy.datetime64 stop_time: Inclusive end of the time window.
        :return: Subset of the measurement dataset for the selected time range.
        """
        return dataset.sel(date_time=slice(start_time, stop_time))


    # def _get_mock_meteo_data(self):
    #     import pandas as pd
    #     import xarray as xr
    #     n_loc = 1
    #     precipitation_flux = -0.0166
    #     time_step = pd.Timedelta(hours=0.025)
    #     lon = [14.41854] * n_loc
    #     lat = [50.073658] * n_loc
    #     start_date = "2025-03-06"
    #     end_date = "2025-03-09"
    #     time = pd.date_range(start=start_date, end=end_date, freq=pd.Timedelta(hours=1))
    #
    #     meteo_data = xr.Dataset(
    #         data_vars=dict(
    #             surface_solar_radiation_downwards=(["loc", "date_time"], np.zeros((n_loc, time.size))),
    #             surface_thermal_radiation_downwards=(["loc", "date_time"], np.zeros((n_loc, time.size))),
    #             precipitation_amount_accum=(["loc", "date_time"], precipitation_flux * np.ones((n_loc, time.size))),
    #             air_temperature_2m=(["loc", "date_time"], 300 * np.ones((n_loc, time.size))),
    #             wind_speed_10m=(["loc", "date_time"], np.zeros((n_loc, time.size))),
    #             wind_from_direction_10m=(["loc", "date_time"], np.zeros((n_loc, time.size))),
    #             air_pressure_at_sea_level=(["loc", "date_time"], 1e5 * np.ones((n_loc, time.size))),
    #             relative_humidity_2m=(["loc", "date_time"], np.zeros((n_loc, time.size))),
    #         ),
    #         coords=dict(
    #             lon=("loc", lon),
    #             lat=("loc", lat),
    #             date_time=time,
    #         ),
    #     )
    #
    #     return meteo_data

    # def get_meteo_for_lon_lat_time(self, start_time, stop_time, longitude, latitude):
    #     """
    #     Retrieve meteorological data for a given location and time window.
    #
    #     :param numpy.datetime64 start_time: Inclusive start of the time window.
    #     :param numpy.datetime64 stop_time: Inclusive end of the time window.
    #     :param float longitude: Target longitude.
    #     :param float latitude: Target latitude.
    #     :return: Meteorological dataset subset for nearest grid point and selected time range.
    #     """
    #
    #     # Select nearest grid point to given coordinates
    #     meteo_lon_lat_dset = self.meteo_dataset.sel(
    #         latitude=latitude,
    #         longitude=longitude,
    #         method="nearest"
    #     )
    #
    #     #meteo_lon_lat_dset = self._get_mock_meteo_data() #@TODO: Use real meteo data ASAP
    #
    #     # Slice dataset by time range
    #     return meteo_lon_lat_dset.sel(date_time=slice(start_time, stop_time))

    def get_long_lat(self, target_time: np.datetime64) -> tuple[float, float]:
        """
        Retrieve longitude and latitude for the site at the specified time.

        :param numpy.datetime64 target_time: Timestamp for which the site coordinates should be retrieved.
        :return: Tuple containing (longitude, latitude) in decimal degrees.
        """

        # Select measurement record at given time
        meas_at_target_time = self.measurements_dataset.sel(date_time=target_time)

        # Extract scalar longitude and latitude values
        longitude = meas_at_target_time.longitude.compute().item()
        latitude = meas_at_target_time.latitude.compute().item()

        return longitude, latitude

    def step(self, start_time, target_time, pressure_at_bottom):
        """
        Advance the 1D model state from start_time to target_time.

        :param numpy.datetime64 start_time: Start of the integration window.
        :param numpy.datetime64 target_time: Target time to which the model advances.
        :param float pressure_at_bottom: Boundary condition (pressure head at bottom).
        :return: Data1DTo3D object containing updated state contribution for 3D model.
        """
        print(f"[1D {self.site_id}] step at t={target_time}")

        # Retrieve spatial coordinates of the site at the target time
        longitude, latitude = self.get_long_lat(target_time)

        # Retrieve measurement data within the time window
        measurements = self.get_dataset_values_for_time(self.measurements_dataset, start_time, target_time)
        # Retrieve meteo data within the time window
        meteo = self.get_dataset_values_for_time(self.meteo_dataset, start_time, target_time)

        # Retrieve meteorological data aligned to location and time window
        #meteo = self.get_meteo_for_lon_lat_time(start_time, target_time, longitude, latitude)

        darcy_velocity = None

        # Perform Kalman update only if measurements are available
        if len(measurements) > 0:
            darcy_velocity = self.kalman.kalman_step(
                self.ukf,
                measurements,
                meteo,
                pressure_at_bottom
            )

        # Package result for downstream 3D model
        return Data1DTo3D(
            date_time=target_time,
            site_id=self.site_id,
            longitude=longitude,
            latitude=latitude,
            velocity=darcy_velocity
        )

    def run_loop(self, t_start, t_end, queue_name_in, queue_name_out):
        """
        Execute the main processing loop for the 1D model using input/output queues.

        :param numpy.datetime64 t_start: Initial simulation time.
        :param numpy.datetime64 t_end: Final simulation time.
        :param str queue_name_in: Name of the input queue receiving 3D→1D data.
        :param str queue_name_out: Name of the output queue sending 1D→3D data.
        :return: Completion message with final state.
        """

        # Initialize input/output queues
        q_in = Queue(queue_name_in)
        q_out = Queue(queue_name_out)

        current_time = t_start

        # Main time-stepping loop
        while current_time < t_end:
            # Blocking call: waits for next 3D→1D message
            target_time, data_to_1d = q_in.get()

            print("current time:{}, target time: {}".format(current_time, target_time))

            # Ensure data consistency (message belongs to this site)
            assert self.site_id == data_to_1d.site_id

            # Perform one model step using received boundary condition
            data_to_3d = self.step(
                current_time,
                target_time,
                data_to_1d.pressure_head
            )

            # Send result back to 3D model
            q_out.put(data_to_3d)

            # Advance internal time
            current_time = target_time

        # Persist Kalman filter results after loop completion
        self.kalman.save_results()

        print(f"[1D {self.idx}] finished loop at t={current_time} (t_end={t_end})")

        return f"1D model {self.idx} done; final state={self.state}"


    @staticmethod
    def calculate_kalman_R_matrix(num_meas_per_probe) -> np.ndarray:
        """
        Construct the measurement noise covariance matrix R for Kalman
        :return: Diagonal covariance matrix of size (num measurements depths, num measurements depths).
        """
        return np.eye(num_meas_per_probe)
