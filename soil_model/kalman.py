import shutil
import sys
import os
import time
from pathlib import Path
import yaml
import argparse
import joblib
import numpy as np
from itertools import groupby
# from joblib import Memory
# memory = Memory(location='cache_dir', verbose=10)
from kalman_result import KalmanResults
from parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
# from soil_model.evapotranspiration_fce import ET0
from auxiliary_functions import sqrt_func, add_noise
from data.load_data import load_data
from kalman_state import StateStructure, MeasurementsStructure
from parallel_ukf import ParallelUKF
from multiprocessing import Manager, Lock
import cloudpickle

######
# Unscented Kalman Filter for Parflow model
# See __main__ for profiling entrypoint.
######


class KalmanFilter:
    """High-level driver for configuring and running a UKF on a ParFlow-based model."""

    @staticmethod
    def from_config(workdir, config_path, verbose=False):
        """
        Create a KalmanFilter from a YAML configuration file.

        :param workdir: Working directory where outputs are written
        :param config_path: Path to YAML configuration file
        :param verbose: Whether to print verbose runtime logs
        :return: Configured KalmanFilter instance
        """
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return KalmanFilter(config_dict, workdir, verbose)

    def __init__(self, config, workdir, verbose=False):
        """
        Initialize the KalmanFilter with model, state, and measurement configs.

        :param config: Dictionary loaded from YAML with model/kalman/measurement sections
        :param workdir: Working directory where outputs are written
        :param verbose: Whether to print verbose runtime logs
        :return: None
        """
        self.work_dir = Path(workdir)
        self.verbose = verbose
        np.random.seed(config["seed"])

        self.kalman_config = config["kalman_config"]
        self.model_config = config["model_config"]
        self.measurements_config = config["measurements_config"]
        self.model = self._make_model()
        nodes_z = self.model.get_nodes_z()

        self.state_struc = StateStructure(len(nodes_z) - 1, self.kalman_config["state_params"])

        self.train_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["train_measurements"])
        self.test_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["test_measurements"])

        # Shared state for multiprocessing UKF
        manager = Manager()
        self.state_measurements = manager.dict()  # (train_meas_dict, test_meas_dict) keyed by encoded state
        self.state_model_velocity_moisture = manager.dict()  # (velocity, moisture) keyed by encoded state
        self.lock = manager.Lock()

        # Expand rainfall schedule into a per-timestep list
        precipitation_list = []
        for (time_prec, precipitation) in self.measurements_config['rain_periods']:
            precipitation_list.extend([precipitation] * time_prec)
        self.measurements_config["precipitation_list"] = precipitation_list

        self.results = KalmanResults(
            workdir, nodes_z, self.state_struc,
            self.train_measurements_struc, self.test_measurements_struc,
            config['postprocess']
        )

    def _make_model(self):
        """
        Build the forward model from configuration.

        :return: Model instance
        """
        if self.model_config["model_class_name"] == "ToyProblem":
            model_class = ToyProblem
        else:
            raise NotImplemented("Import desired class")
        return model_class(self.model_config, workdir=self.work_dir / "output-toy")

    def process_loaded_measurements(self, noisy_measurements_train, noisy_measurements_test, measurement_state_flag):
        """
        Align preloaded measurements with precipitation schedule and iteration grouping.

        :param noisy_measurements_train: Encoded training measurements list/array
        :param noisy_measurements_test: Encoded testing measurements list/array
        :param measurement_state_flag: Measurements state flag list/array
        :return: Tuple (noisy_train_measurements, noisy_test_measurements,
            meas_model_iter_time, meas_model_iter_flux)
        """
        total_time = len(self.measurements_config["precipitation_list"])
        print("total time ", total_time)
        meas_model_iter_time = []
        meas_model_iter_flux = []
        noisy_train_measurements = []
        noisy_test_measurements = []
        measurement_state_flag_sampled = []

        print("len(noisy_measurements_train) ", len(noisy_measurements_train))
        total_index = 0
        step = int(self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"])
        for i in range(0, int(total_time), step):
            precipitation_step_start = i
            precipitation_step_end = np.min([i + step, int(total_time)])
            print("prec start: {}, end: {}".format(precipitation_step_start, precipitation_step_end))

            # Group by consecutive equal precipitation flux over this window
            window = self.measurements_config["precipitation_list"][precipitation_step_start:precipitation_step_end]
            prec_time_flux_per_iter = [(len(list(n_times)), flux) for flux, n_times in groupby(window)]
            print("prec_time_flux_per_iter ", prec_time_flux_per_iter)

            for (prec_time, prec_flux) in prec_time_flux_per_iter:
                measurements_time_step = self.measurements_config["measurements_time_step"]
                n_time_steps_per_iteration = prec_time / measurements_time_step
                print("n_time_steps_per_iteration ", n_time_steps_per_iteration)

                total_index += int(n_time_steps_per_iteration)

                try:
                    print("noisy_measurements_train[total_index])", noisy_measurements_train[total_index])
                    print("noisy_measurements_test[total_index])", noisy_measurements_test[total_index])

                    noisy_train_measurements.append(noisy_measurements_train[total_index])
                    noisy_test_measurements.append(noisy_measurements_test[total_index])
                    measurement_state_flag_sampled.append(measurement_state_flag[total_index])
                except IndexError as idxerr:
                    print("idx_error ", idxerr)
                    noisy_train_measurements.append(noisy_measurements_train[total_index - 1])
                    noisy_test_measurements.append(noisy_measurements_test[total_index - 1])
                    measurement_state_flag_sampled.append(measurement_state_flag[total_index - 1])

                meas_model_iter_time.append(prec_time)
                meas_model_iter_flux.append(prec_flux)

        return noisy_train_measurements, noisy_test_measurements, measurement_state_flag_sampled, meas_model_iter_time, meas_model_iter_flux

    def run(self):
        """
        Run the UKF: load/generate measurements, configure filter, and iterate.

        Uses either pre-recorded measurements (CSV) or synthetic ones from the model.

        :return: KalmanResults instance with full time series of states, covariances, and plots metadata
        """
        #############################
        ### Generate measurements ###
        #############################
        if "measurements_file" in self.measurements_config:
            noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux, measurement_state_flag = load_data(
                self.train_measurements_struc,
                self.test_measurements_struc,
                data_csv=self.measurements_config["measurements_file"],
                measurements_config=self.measurements_config
            )

            precipitation_list = []
            for (time_prec, precipitation) in meas_model_iter_flux:
                precipitation_list.extend([precipitation] * time_prec)
            self.measurements_config["precipitation_list"] = precipitation_list

            noisy_measurements, noisy_measurements_to_test, measurement_state_flag_sampled, meas_model_iter_time, meas_model_iter_flux = \
                self.process_loaded_measurements(noisy_measurements, noisy_measurements_to_test, measurement_state_flag)

            sample_variance = np.nanvar(noisy_measurements, axis=0)
            measurement_noise_covariance = np.diag(sample_variance)

            self.results.times_measurements = np.cumsum(meas_model_iter_time)
        else:
            # Generate synthetic measurements via forward model runs
            measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, \
            state_data_iters, meas_model_iter_time, meas_model_iter_flux = self.generate_measurements()

            measurement_state_flag = []  # No state flag for synthetic data
            residuals = noisy_measurements - measurements
            measurement_noise_covariance = np.cov(residuals, rowvar=False)

            self.results.ref_states = np.array(state_data_iters)
            self.results.train_measuremnts_exact = measurements
            self.results.test_measuremnts_exact = measurements_to_test
            print("meas_model_iter_time ", meas_model_iter_time)
            self.results.times_measurements = np.cumsum(meas_model_iter_time)
            print("self.results.times_measurements ", self.results.times_measurements)

        self.results.precipitation_flux_measurements = meas_model_iter_flux

        #######################################
        ### UKF settings: sigma points, Q/R ###
        #######################################
        ukf = self.set_kalman_filter(measurement_noise_covariance)

        #######################################
        ### UKF run: predict / update loop  ###
        #######################################
        self.run_kalman_filter(ukf, noisy_measurements, measurement_state_flag_sampled)

        return self.results

    def model_run(self, flux, stop_time, time_step, pressure, params):
        """
        Execute the forward model for a single interval.

        :param flux: Precipitation flux for the interval
        :param stop_time: End time for the run (relative)
        :param time_step: Model time step
        :param pressure: Initial pressure field
        :param params: State parameters dictionary (decoded)
        :return: New pressure field at the end of the run
        """
        self.model.run(
            init_pressure=pressure, precipitation_value=flux,
            state_params=params, start_time=0, stop_time=stop_time, time_step=time_step
        )
        new_pressure = self.model.get_data(current_time_step=stop_time, data_name="pressure")
        return new_pressure

    def model_iteration(self, precipitation_flux, pressure, params, model_time_step, model_n_time_steps_per_iter):
        """
        Advance the model for one precipitation segment and collect outputs.

        :param precipitation_flux: Flux value for this iteration
        :param pressure: Current pressure field (initial condition)
        :param params: Decoded state parameters dict
        :param model_time_step: Model time step
        :param model_n_time_steps_per_iter: Number of steps to run in this iteration
        :return: Tuple (train_measurements, test_measurements, new_pressure, new_saturation)
        """
        et_per_time = 0  # placeholder for ET implementation
        stop_time = model_time_step * model_n_time_steps_per_iter

        print("model stop time ", stop_time)
        print("model time step ", model_time_step)

        new_pressure = self.model_run(precipitation_flux, stop_time, model_time_step, pressure, params)

        new_saturation = self.model.get_data(current_time_step=stop_time, data_name="moisture")
        measurements_train = self.get_measurement(current_time_step=stop_time, measurements_struct=self.train_measurements_struc)
        print("measurements_train ", measurements_train)
        measurements_test = self.get_measurement(current_time_step=stop_time, measurements_struct=self.test_measurements_struc)

        return measurements_train, measurements_test, new_pressure, new_saturation

    def generate_measurements(self):
        """
        Generate synthetic train/test measurements by running the forward model.

        :return: Tuple (train_measurements, noisy_train_measurements,
            test_measurements, noisy_test_measurements,
            state_data_iters, meas_model_iter_time, meas_model_iter_flux)
        """
        train_measurements = []
        test_measurements = []
        noisy_train_measurements = []
        noisy_test_measurements = []
        state_data_iters = []

        # Initial state / reference parameters
        pressure_vec = self.model.make_linear_pressure(self.model_config)
        ref_params = self.state_struc.compose_ref_dict()
        ref_params['pressure_field'] = pressure_vec
        total_time = len(self.measurements_config["precipitation_list"])
        print("total time ", total_time)
        meas_model_iter_time = []
        meas_model_iter_flux = []

        step = int(self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"])
        for i in range(0, int(total_time), step):
            precipitation_step_start = i
            precipitation_step_end = np.min([i + step, int(total_time)])
            print("prec start: {}, end: {}".format(precipitation_step_start, precipitation_step_end))
            model_time_step = self.measurements_config["model_time_step"]

            window = self.measurements_config["precipitation_list"][precipitation_step_start:precipitation_step_end]
            prec_time_flux_per_iter = [(len(list(n_times)), flux) for flux, n_times in groupby(window)]

            print("prec_time_flux_per_iter ", prec_time_flux_per_iter)

            for (prec_time, prec_flux) in prec_time_flux_per_iter:
                model_n_time_steps_per_iteration = prec_time / model_time_step

                measurement_train, measurement_test, pressure_vec, sat_vec = self.model_iteration(
                    prec_flux, pressure_vec, ref_params,
                    model_time_step=model_time_step,
                    model_n_time_steps_per_iter=model_n_time_steps_per_iteration
                )
                self.results.ref_saturation.append(sat_vec)

                train_measurements.append(self.train_measurements_struc.encode(measurement_train))
                test_measurements.append(self.test_measurements_struc.encode(measurement_test))

                calibration_coeffs_z_positions = self.state_struc.get_calibration_coeffs_z_positions()
                if len(calibration_coeffs_z_positions) > 0:
                    measurement_train = self.train_measurements_struc.mult_calibration_coef(
                        self.train_measurements_struc,
                        measurement_train,
                        ref_params["calibration_coeffs"],
                        np.squeeze(calibration_coeffs_z_positions)
                    )

                noisy_train_measurements.append(
                    self.train_measurements_struc.encode(measurement_train, noisy=True)
                )
                noisy_test_measurements.append(
                    self.test_measurements_struc.encode(measurement_test, noisy=True)
                )

                if self.verbose:
                    print("i: {}, data_pressure: {} ".format(i, pressure_vec))
                ref_params['pressure_field'] = pressure_vec

                iter_state = self.state_struc.encode_state(ref_params)
                state_data_iters.append(iter_state)

                meas_model_iter_time.append(prec_time)
                meas_model_iter_flux.append(prec_flux)

        train_measurements = np.array(train_measurements)
        test_measurements = np.array(test_measurements)
        noisy_train_measurements = np.array(noisy_train_measurements)
        noisy_test_measurements = np.array(noisy_test_measurements)

        return (train_measurements, noisy_train_measurements, test_measurements, noisy_test_measurements,
                state_data_iters, meas_model_iter_time, meas_model_iter_flux)

    def get_measurement(self, current_time_step, measurements_struct):
        """
        Sample model outputs at sensor locations defined by a MeasurementsStructure.

        :param current_time_step: Time (in model units) at which to sample
        :param measurements_struct: MeasurementsStructure defining sensors and interpolation
        :return: Dictionary mapping measurement names to arrays of sampled values
        """
        measurements_dict = {}
        for measurement_name, measure_obj in measurements_struct.items():
            data_to_measure = self.model.get_data(current_time_step=current_time_step, data_name=measurement_name)
            measurements_dict[measurement_name] = measure_obj.interp @ data_to_measure
        return measurements_dict

    #####################
    ### Kalman filter ###
    #####################
    def state_transition_function(self, state_vec, dt, iter_duration, precipitation_flux):
        """
        UKF state transition function: advance model state over one iteration.

        Spawns a unique ParFlow working directory per call (safe for multiprocessing).

        :param state_vec: Encoded current state vector
        :param dt: UKF dt parameter (not used directly when iter_duration provided)
        :param iter_duration: Physical duration of this iteration (model time)
        :param precipitation_flux: Precipitation flux applied during this iteration
        :return: Encoded next state vector
        """
        print("dt: ", dt, "iter duration time: ", iter_duration)
        pid = os.getpid()
        timestamp = int(time.time())

        if os.environ.get("SCRATCHDIR"):
            scratch_dir = os.environ.get("SCRATCHDIR")
            parflow_working_dir = os.path.join(scratch_dir, f"parflow_working_dir_{pid}_{timestamp}")
        else:
            parflow_working_dir = os.path.join(self.model._workdir, f"parflow_working_dir_{pid}_{timestamp}")
        os.makedirs(parflow_working_dir)

        state = self.state_struc.decode_state(state_vec)
        pressure_data = state["pressure_field"]

        et_per_time = 0  # placeholder for ET computation
        iter_duration = float(iter_duration)

        self.model.run(
            init_pressure=pressure_data, precipitation_value=precipitation_flux,
            state_params=state, start_time=0, stop_time=iter_duration,
            time_step=self.kalman_config["model_time_step"],
            working_dir=parflow_working_dir
        )

        state["pressure_field"] = self.model.get_data(current_time_step=iter_duration, data_name="pressure")

        velocity = self.model.get_data(current_time_step=iter_duration, data_name="velocity")
        moisture = self.model.get_data(current_time_step=iter_duration, data_name="moisture")

        measurements_train = self.get_measurement(current_time_step=iter_duration,
                                                  measurements_struct=self.train_measurements_struc)
        measurements_test = self.get_measurement(current_time_step=iter_duration,
                                                 measurements_struct=self.test_measurements_struc)
        new_state_vec = self.state_struc.encode_state(state)

        if self.lock:
            self.state_measurements[tuple(new_state_vec)] = (measurements_train, measurements_test)
            self.state_model_velocity_moisture[tuple(new_state_vec)] = (velocity, moisture)

        shutil.rmtree(parflow_working_dir)
        return new_state_vec

    def measurement_function(self, state_vec, measurements_type="train"):
        """
        UKF measurement function: map state to predicted sensor observations.

        :param state_vec: Encoded state vector whose predictions are requested
        :param measurements_type: 'train' or 'test' to choose the target structure
        :return: Encoded measurement vector for the requested structure
        """
        measurements_train_dict, measurements_test_dict = self.state_measurements[tuple(state_vec)]

        if measurements_type == "train":
            calibration_coeffs_z_positions = self.state_struc.get_calibration_coeffs_z_positions()
            if len(calibration_coeffs_z_positions) > 0:
                measurements_train = self.train_measurements_struc.mult_calibration_coef(
                    self.train_measurements_struc,
                    measurements_train_dict,
                    self.state_struc.decode_state(state_vec)["calibration_coeffs"],
                    np.squeeze(calibration_coeffs_z_positions)
                )
                return self.train_measurements_struc.encode(
                    measurements_train, state=self.state_struc.decode_state(state_vec)
                )
            else:
                return self.train_measurements_struc.encode(
                    measurements_train_dict, state=self.state_struc.decode_state(state_vec)
                )
        elif measurements_type == "test":
            return self.test_measurements_struc.encode(
                measurements_test_dict, state=self.state_struc.decode_state(state_vec)
            )

    @staticmethod
    def get_sigma_points_obj(sigma_points_params, num_state_params):
        """
        Build the sigma points object for the UKF.

        :param sigma_points_params: Dict of alpha/beta/kappa (and other filterpy kwargs)
        :param num_state_params: Dimension of the state vector
        :return: Sigma points object compatible with filterpy UKF
        """
        return MerweScaledSigmaPoints(n=num_state_params, sqrt_method=sqrt_func, **sigma_points_params)

    def add_noise_to_init_state(self, init_state, init_pressure_data):
        """
        Add noise to the initial decoded state (pressure and other parameters).

        :param init_state: Decoded initial state dictionary
        :param init_pressure_data: Initial pressure field before noise
        :return: Decoded initial state dictionary with noise applied
        """
        for key in self.state_struc.keys():
            if key in init_state:
                if key == "pressure_field":
                    init_state["pressure_field"] = add_noise(
                        np.squeeze(init_pressure_data),
                        noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                        distr_type=self.kalman_config["noise_distr_type"]
                    )
                else:
                    noisy_param_value = self.state_struc[key].transform_from_gauss(
                        add_noise(
                            np.array([self.state_struc[key].transform_to_gauss(init_state[key])]),
                            noise_level=self.state_struc[key].std,
                            distr_type=self.kalman_config["noise_distr_type"]
                        )
                    )
                    init_state[key] = np.squeeze(noisy_param_value)
        return init_state

    def set_kalman_filter(self, measurement_noise_covariance):
        """
        Configure the UKF object: dimensions, sigma points, Q, R, initial x and P.

        :param measurement_noise_covariance: Measurement noise covariance matrix (R)
        :return: Configured UKF/ParallelUKF instance
        """
        num_state_params = self.state_struc.size()
        dim_z = measurement_noise_covariance.shape[0]

        sigma_points_params = self.kalman_config["sigma_points_params"]
        sigma_points = KalmanFilter.get_sigma_points_obj(sigma_points_params, num_state_params)

        time_step = 1  # UKF internal dt (hours) — physical duration passed via kwargs

        ukf = ParallelUKF(
            dim_x=num_state_params, dim_z=dim_z, dt=time_step,
            fx=self.state_transition_function, hx=self.measurement_function,
            points=sigma_points
        )

        Q_state = self.state_struc.compose_Q()
        ukf.Q = Q_state
        print("ukf.Q.shape ", ukf.Q.shape)
        print("ukf.Q ", ukf.Q)
        print("diag ukf.Q ", np.diag(ukf.Q))
        ukf.R = measurement_noise_covariance
        print("R measurement_noise_covariance ", measurement_noise_covariance)

        print("self.model ", self.model)

        data_pressure = self.model.make_linear_pressure(self.model_config)
        print("data pressure ", data_pressure)

        el_centers_z = self.model.get_el_centers_z()
        init_mean, init_cov = self.state_struc.compose_init_state(el_centers_z)

        init_state = self.state_struc.decode_state(init_mean)
        init_state = self.add_noise_to_init_state(init_state, data_pressure)

        ukf.x = self.state_struc.encode_state(init_state)

        init_cov_multiplicator = self.kalman_config.get("init_cov_P_multiplicator", 1)
        ukf.P = init_cov * init_cov_multiplicator

        print("init cov ", init_cov.shape)
        print("np.diag(init_cov) ", np.diag(init_cov))

        return ukf

    def run_kalman_filter(self, ukf, noisy_measurements, measurement_state_flag):
        """
        Run the UKF predict/update loop for all measurement timesteps.

        :param ukf: Configured UKF/ParallelUKF instance
        :param noisy_measurements: Sequence of encoded measurement vectors
        :param measurement_state_flag: Optional flags to skip updates at given steps
        :return: KalmanResults with stored time series
        """
        iter_durations = [self.results.times_measurements[0]] + list(
            np.array(self.results.times_measurements[1:]) - np.array(self.results.times_measurements[:-1])
        )
        print("iter durations ", iter_durations)
        print("noisy_measurements ", noisy_measurements)

        for i, measurement in enumerate(noisy_measurements):
            ukf.predict(iter_duration=iter_durations[i], precipitation_flux=self.results.precipitation_flux_measurements[i])
            print("i: {}, measurement: {} ".format(i, measurement))

            # Skip bad measurements if flagged
            if i < len(measurement_state_flag) and measurement_state_flag[i] != 0:
                print(f"[UKF] Skipping update at timestep {i} (bad measurements)")
            else:
                ukf.update(measurement)

            print("sum ukf.P ", np.sum(ukf.P))
            print("Estimated State:", ukf.x)

            self.results.times.append(self.results.times_measurements[i])
            self.results.ukf_x.append(ukf.x)
            self.results.ukf_P.append(ukf.P)
            self.results.measurement_in.append(measurement)

            velocity, moisture = self.state_model_velocity_moisture[list(self.state_measurements.keys())[-1]]
            self.results.velocities.append(velocity)
            self.results.moistures.append(moisture)

            measurements_train_dict, measurements_test_dict = self.state_measurements[list(self.state_measurements.keys())[-1]]
            self.results.ukf_train_meas.append(self.train_measurements_struc.encode(measurements_train_dict))
            self.results.ukf_test_meas.append(self.test_measurements_struc.encode(measurements_test_dict))

        joblib.dump(self.results, self.work_dir / 'kalman_results.pkl')
        return self.results


# @memory.cache
def run_kalman(workdir, cfg_file):
    """
    Convenience function to run the full UKF pipeline and return results.

    :param workdir: Working directory
    :param cfg_file: Path to YAML configuration file
    :return: KalmanResults instance
    """
    kalman_filter = KalmanFilter.from_config(Path(workdir), Path(cfg_file).resolve(), verbose=False)
    return kalman_filter.run()


def main():
    """
    CLI entry point for running the UKF and postprocessing.
    Expects two arguments:
      1) work_dir: path to working directory
      2) config_file: path to YAML config

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Path to work dir')
    parser.add_argument('config_file', help='Path to configuration file')
    args = parser.parse_args(sys.argv[1:])
    results = run_kalman(Path(args.work_dir), Path(args.config_file).resolve())
    results.postprocess()


if __name__ == "__main__":
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    main()

    # Configure ParFlow executable paths if needed
    # os.environ['PARFLOW_HOME'] = '/opt/parflow_install'
    # os.environ['PATH'] += ':/opt/parflow_install/bin'

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)
