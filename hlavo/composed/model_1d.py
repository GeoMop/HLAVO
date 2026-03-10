from dask.distributed import Queue
from hlavo.kalman.kalman import KalmanFilter
from hlavo.ingress.moist_profile.load_data import load_pr2_data, load_odyssey_data, preprocess_data, get_measurements, get_precipitations, load_data
from hlavo.ingress.moist_profile.load_zarr_data import load_zarr_data
from bisect import bisect_left, bisect_right

# ---------------------------------------------------------------------------
# 1D model class
# ---------------------------------------------------------------------------

class Model1D:
    def __init__(self, idx, initial_state=0.0, work_dir=None, kalman_config_path=None, seed=None):
        self.idx = idx
        self.state = initial_state

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
        noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux, measurement_state_flag, timestamps = load_zarr_data(
            self.kalman.train_measurements_struc,
            self.kalman.test_measurements_struc,
            zarr_dir=self.kalman.measurements_config["zarr_dir"],
            scheme_file=self.kalman.measurements_config["scheme_file"],
            measurements_config=self.kalman.measurements_config
        )

        exit()



        noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux, measurement_state_flag, timestamps = load_data(
            self.kalman.train_measurements_struc,
            self.kalman.test_measurements_struc,
            data_csv=self.kalman.measurements_config["measurements_file"],
            measurements_config=self.kalman.measurements_config
        )

        precipitation_list = []
        for (time_prec, precipitation) in meas_model_iter_flux:
            precipitation_list.extend([precipitation] * time_prec)
        self.kalman.measurements_config["precipitation_list"] = precipitation_list

        noisy_measurements, noisy_measurements_to_test, measurement_state_flag_sampled, meas_model_iter_time, meas_model_iter_flux, meas_model_iter_timestamps = \
            self.kalman.process_loaded_measurements(noisy_measurements, noisy_measurements_to_test, measurement_state_flag, timestamps)

        sample_variance = np.nanvar(noisy_measurements, axis=0)
        measurement_noise_covariance = np.diag(sample_variance)

        self.measurements = noisy_measurements
        self.measurement_state_flag_sampled = measurement_state_flag_sampled
        self.measurements_timestamps = meas_model_iter_timestamps
        self.precipitation_flux_measurements = meas_model_iter_flux

        assert len(self.measurements) == len(self.measurements_timestamps) == len(self.measurement_state_flag_sampled) == len(self.precipitation_flux_measurements)

        self.kalman.results.times_measurements = np.cumsum(meas_model_iter_time)
        self.kalman.results.precipitation_flux_measurements = meas_model_iter_flux

        return self.kalman.set_kalman_filter(measurement_noise_covariance)


    def get_measurement_for_time(self, start_time, stop_time):
        """
        Retrieve all measurements and flags within a time window:
            start_time <= timestamp <= stop_time

        Uses binary search (bisect) for O(log n) boundary lookup.
        Assumes:
            - self.measurements_timestamps is sorted
            - timestamps are datetime objects
        """

        i0 = bisect_left(self.measurements_timestamps, start_time)
        i1 = bisect_right(self.measurements_timestamps, stop_time)

        selected_measurements = self.measurements[i0:i1]
        selected_measurements_flags = self.measurement_state_flag_sampled[i0:i1]

        return selected_measurements, selected_measurements_flags


    def get_precipitation_for_time(self, start_time, stop_time):
        """
        Retrieve precipitation flux values within the given time window.
        Uses the same slicing logic as measurement retrieval to ensure
        temporal alignment.
        """
        i0 = bisect_left(self.measurements_timestamps, start_time)
        i1 = bisect_right(self.measurements_timestamps, stop_time)
        return self.kalman.results.precipitation_flux_measurements[i0:i1]


    def step(self, start_time, target_time, pressure_at_bottom):
        """
        Advance the 1D model state from start_time to target_time.
        Steps:
        1. Update physical state using provided forcing data.
        2. Retrieve measurements within this time window.
        3. Retrieve aligned precipitation flux.
        4. Execute one Kalman step.
        """
        print(f"[1D {self.idx}] step at t={target_time}")

        # measurement must come from somewhere meaningful
        measurements, measurements_flag = self.get_measurement_for_time(start_time, target_time)
        precipitation_flux = self.get_precipitation_for_time(start_time, target_time)

        darcy_velocity = None
        if len(measurements) > 0:
            darcy_velocity = self.kalman.kalman_step(self.ukf, start_time, target_time, measurements,
                                               measurements_flag, precipitation_flux, pressure_at_bottom)

        return darcy_velocity

    def run_loop(self, t_start, t_end, queue_name_in, queue_name_out):
        """
        Input queue processing loop for the 1D model.
        """
        #client = get_client()
        q_in = Queue(queue_name_in)       # correct API
        q_out = Queue(queue_name_out)

        current_time = t_start

        while current_time < t_end:
            target_time, data = q_in.get() # How target time gets into Queue?    # blocks
            print("current time:{}, target time: {}".format(current_time, target_time))

            contribution = self.step(current_time, target_time, data) #@TODO: Why we need contribution?
            q_out.put((self.idx, target_time, contribution))
            print(f"[1D {self.idx}] sent contribution={contribution} at t={target_time}")

            current_time = target_time

        self.kalman.save_results()
        print(f"[1D {self.idx}] finished loop at t={current_time} (t_end={t_end})")
        return f"1D model {self.idx} done; final state={self.state}"
