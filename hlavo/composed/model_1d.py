import numpy as np
from dask.distributed import Queue
from hlavo.kalman.kalman import KalmanFilter
from hlavo.ingress.moist_profile.load_data import load_pr2_data, load_odyssey_data, preprocess_data, get_measurements, get_precipitations, load_data
from hlavo.ingress.moist_profile.load_zarr_data import load_zarr_data
from bisect import bisect_left, bisect_right
from hlavo.composed.data_1d_to_3d import Data1DTo3D

# ---------------------------------------------------------------------------
# 1D model class
# ---------------------------------------------------------------------------

class Model1D:
    def __init__(self, site_id, initial_state=0.0, work_dir=None, kalman_config_path=None, seed=None):
        self.site_id = site_id
        self.state = initial_state

        self.measurements_dataset = None

        print("workdir ", work_dir)
        print("kalman config path ", kalman_config_path)

        # work_dir.mkdir(parents=True, exist_ok=True)
        # shutil.copy(kalman_config_path, work_dir / kalman_config_path.name)
        #
        # kalman_config_path = work_dir / kalman_config_path.name
        #
        # print("final kalman config path ", kalman_config_path)

        self.kalman = KalmanFilter.from_config(work_dir, kalman_config_path, verbose=False, seed=seed)
        self.ukf = self._prepare_kalman_measurements()

        #kalman_filter.run() # load measurements

    def _prepare_kalman_measurements(self):
        """
        Load, preprocess, and initialize all measurement-related data
        required for the Kalman filter.

        This method:
        1. Loads raw measurement data from CSV.
        2. Expands precipitation flux into per-step values.
        3. Applies internal Kalman preprocessing/sampling.
        4. Computes measurement noise covariance from sample variance.
        5. Stores aligned arrays used during runtime slicing.
        6. Initializes the UKF with estimated measurement covariance.
        """
        print("self.kalman.measurements_config ", self.kalman.measurements_config)
        measurements_xarray = load_zarr_data(
            self.kalman.train_measurements_struc,
            self.kalman.test_measurements_struc,
            zarr_dir=self.kalman.measurements_config["zarr_dir"],
            scheme_file=self.kalman.measurements_config["scheme_file"],
            measurements_config=self.kalman.measurements_config)

        self.measurements_dataset = measurements_xarray.sel(site_id=self.site_id)

        print("type(self.measurements_dataset) ", type(self.measurements_dataset))

        #@TODO: how to calculate ukf.R - meas covariance mat?

        moisture_meas = measurements_xarray["moisture"]

        print("moisture_meas ", moisture_meas)

        print("measurements_xarray.latitude ", measurements_xarray.latitude)



        # noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux, measurement_state_flag, timestamps = load_data(
        #     self.kalman.train_measurements_struc,
        #     self.kalman.test_measurements_struc,
        #     data_csv=self.kalman.measurements_config["measurements_file"],
        #     measurements_config=self.kalman.measurements_config
        # )
        #
        # precipitation_list = []
        # for (time_prec, precipitation) in meas_model_iter_flux:
        #     precipitation_list.extend([precipitation] * time_prec)
        # self.kalman.measurements_config["precipitation_list"] = precipitation_list
        #
        # noisy_measurements, noisy_measurements_to_test, measurement_state_flag_sampled, meas_model_iter_time, meas_model_iter_flux, meas_model_iter_timestamps = \
        #     self.kalman.process_loaded_measurements(noisy_measurements, noisy_measurements_to_test, measurement_state_flag, timestamps)
        #
        # sample_variance = np.nanvar(noisy_measurements, axis=0)
        # measurement_noise_covariance = np.diag(sample_variance)
        #
        # self.measurements = noisy_measurements
        # self.measurement_state_flag_sampled = measurement_state_flag_sampled
        # self.measurements_timestamps = meas_model_iter_timestamps
        # self.precipitation_flux_measurements = meas_model_iter_flux
        #
        # assert len(self.measurements) == len(self.measurements_timestamps) == len(self.measurement_state_flag_sampled) == len(self.precipitation_flux_measurements)
        #
        # self.kalman.results.times_measurements = np.cumsum(meas_model_iter_time)
        # self.kalman.results.precipitation_flux_measurements = meas_model_iter_flux

        kalman_R_matrix = Model1D.calculate_kalman_R_matrix(len(self.measurements_dataset["depth_level"]))

        return self.kalman.set_kalman_filter(kalman_R_matrix)

    def get_measurement_for_time(self, start_time: np.datetime64, stop_time: np.datetime64):
        """
        Retrieve measurements within a specified time window.

        :param numpy.datetime64 start_time: Inclusive start of the time window.
        :param numpy.datetime64 stop_time: Inclusive end of the time window.
        :return: Subset of the measurement dataset containing all variables for the selected time range.
        """
        return self.measurements_dataset.sel(date_time=slice(start_time, stop_time))

    def get_precipitation_for_time(self, start_time, stop_time):
        """
        Retrieve precipitation flux values within the given time window.
        Uses the same slicing logic as measurement retrieval to ensure
        temporal alignment.
        """
        # i0 = bisect_left(self.measurements_timestamps, start_time)
        # i1 = bisect_right(self.measurements_timestamps, stop_time)

        return self.kalman.measurements_config["precipitation_list"] #@TODO: RM ASAP

        #return self.kalman.results.precipitation_flux_measurements[i0:i1]

    def get_long_lat(self, target_time: np.datetime64) -> tuple[float, float]:
        """
        Retrieve longitude and latitude for the site at the specified time.

        :param numpy.datetime64 target_time: Timestamp for which the site coordinates should be retrieved.
        :return: Tuple containing (longitude, latitude) in decimal degrees.
        """

        meas_at_target_time = self.measurements_dataset.sel(date_time=target_time)

        longitude = meas_at_target_time.longitude.compute().item()
        latitude = meas_at_target_time.latitude.compute().item()

        return longitude, latitude


    def step(self, start_time, target_time, pressure_at_bottom):
        """
        Advance the 1D model state from start_time to target_time.
        Steps:
        1. Update physical state using provided forcing data.
        2. Retrieve measurements within this time window.
        3. Retrieve aligned precipitation flux.
        4. Execute one Kalman step.
        """
        print(f"[1D {self.site_id}] step at t={target_time}")

        # measurement must come from somewhere meaningful
        measurements = self.get_measurement_for_time(start_time, target_time)
        precipitation_flux = self.get_precipitation_for_time(start_time, target_time)[:measurements.sizes["date_time"]] #@TODO: refactor

        darcy_velocity = None
        if len(measurements) > 0:
            darcy_velocity = self.kalman.kalman_step(self.ukf, measurements, precipitation_flux, pressure_at_bottom)

        longitude, latitude = self.get_long_lat(target_time)

        return Data1DTo3D(date_time=target_time, site_id=self.site_id,
                   longitude=longitude, latitude=latitude, velocity=darcy_velocity)


    def run_loop(self, t_start, t_end, queue_name_in, queue_name_out):
        """
        Input queue processing loop for the 1D model.
        """
        #client = get_client()
        q_in = Queue(queue_name_in)       # correct API
        q_out = Queue(queue_name_out)

        current_time = t_start

        while current_time < t_end:
            target_time, data_to_1d = q_in.get() # How target time gets into Queue?    # blocks
            print("current time:{}, target time: {}".format(current_time, target_time))

            assert self.site_id == data_to_1d.site_id

            data_to_3d = self.step(current_time, target_time, data_to_1d.pressure_head) #@TODO: Why we need contribution?
            q_out.put(data_to_3d)
            #print(f"[1D {self.idx}] sent contribution={contribution} at t={target_time}")

            current_time = target_time

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
