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
#from joblib import Memory
#memory = Memory(location='cache_dir', verbose=10)
from kalman_result import KalmanResults
from parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
#from soil_model.evapotranspiration_fce import ET0
from auxiliary_functions import sqrt_func, add_noise
from data.load_data import load_data
from kalman_state import StateStructure, MeasurementsStructure
from parallel_ukf import ParallelUKF
from multiprocessing import Manager, Lock
import cloudpickle

######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######


class KalmanFilter:
    @staticmethod
    def from_config(workdir, config_path, verbose=False):
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return KalmanFilter(config_dict, workdir, verbose)

    def __init__(self, config, workdir, verbose=False):
        self.work_dir = Path(workdir)
        self.verbose = verbose
        np.random.seed(config["seed"])

        self.kalman_config = config["kalman_config"]
        self.model_config = config["model_config"]
        self.measurements_config = config["measurements_config"]
        self.model = self._make_model()
        nodes_z = self.model.get_nodes_z()

        self.state_struc = StateStructure(len(nodes_z) - 1, self.kalman_config["state_params"])

        #print("kalman config meas ", self.kalman_config["train_measurements"])
        self.train_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["train_measurements"])
        self.test_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["test_measurements"])

        manager = Manager()
        self.state_measurements = manager.dict()  # Shared across processes
        self.lock = manager.Lock()  # Lock to avoid race conditions

        precipitation_list = []
        for (time_prec, precipitation) in self.measurements_config['rain_periods']:
            precipitation_list.extend([precipitation] * time_prec)
        self.measurements_config["precipitation_list"] = precipitation_list

        self.results = KalmanResults(workdir, nodes_z, self.state_struc, self.train_measurements_struc,
                                     self.test_measurements_struc, config['postprocess'])

    def _make_model(self):
        if self.model_config["model_class_name"] == "ToyProblem":
            model_class = ToyProblem
        else:
            raise NotImplemented("Import desired class")
        return model_class(self.model_config, workdir=self.work_dir / "output-toy")

    def process_loaded_measurements(self, noisy_measurements_train, noisy_measurements_test):
        total_time = len(self.measurements_config["precipitation_list"])
        print("total time ", total_time)
        meas_model_iter_time = []
        meas_model_iter_flux = []
        noisy_train_measurements = []
        noisy_test_measurements = []

        print("len(noisy_measurements_train) ", len(noisy_measurements_train))
        total_index = 0
        for i in range(0, int(total_time), int(self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"])):
            precipitation_step_start = i
            precipitation_step_end = np.min([i + int(
                self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"]),
                                             int(total_time)])
            print("prec start: {}, end: {}".format(precipitation_step_start, precipitation_step_end))
            model_time_step = self.measurements_config["model_time_step"]
            prec_time_flux_per_iter = [(len(list(n_times)), flux) for flux, n_times in groupby(
                self.measurements_config["precipitation_list"][precipitation_step_start:precipitation_step_end])]

            print("prec_time_flux_per_iter ", prec_time_flux_per_iter)

            for (prec_time, prec_flux) in prec_time_flux_per_iter:
                measurements_time_step = self.measurements_config["measurements_time_step"]
                n_time_steps_per_iteration = prec_time / measurements_time_step
                print("n_time_steps_per_iteration ", n_time_steps_per_iteration)

                total_index += int(n_time_steps_per_iteration)

                print("total index ", total_index)

                try:
                    print("noisy_measurements_train[total_index])", noisy_measurements_train[total_index])
                    print("noisy_measurements_test[total_index])", noisy_measurements_test[total_index])

                    noisy_train_measurements.append(noisy_measurements_train[total_index])
                    noisy_test_measurements.append(noisy_measurements_test[total_index])
                except IndexError as idxerr:
                    print("idx_error ", idxerr)
                    noisy_train_measurements.append(noisy_measurements_train[total_index-1])
                    noisy_test_measurements.append(noisy_measurements_test[total_index-1])

                # continue
                #
                # model_n_time_steps_per_iteration = model_n_time_steps_per_iteration
                # measurement_train, measurement_test, pressure_vec, sat_vec \
                #     = self.model_iteration(prec_flux, pressure_vec, ref_params, model_time_step=model_time_step,
                #                            model_n_time_steps_per_iter=model_n_time_steps_per_iteration)
                # self.results.ref_saturation.append(sat_vec)
                #
                # # print("ref params ", ref_params)
                #
                # train_measurements.append(self.train_measurements_struc.encode(measurement_train))
                # test_measurements.append(self.test_measurements_struc.encode(measurement_test))
                #
                # # print("self.train_measurements_struc.z_positions ", self.train_measurements_struc.z_positions())
                #
                # calibration_coeffs_z_positions = self.state_struc.get_calibration_coeffs_z_positions()
                # if len(calibration_coeffs_z_positions) > 0:
                #     measurement_train = self.train_measurements_struc.mult_calibration_coef(
                #         self.train_measurements_struc, measurement_train, ref_params["calibration_coeffs"],
                #         np.squeeze(calibration_coeffs_z_positions))
                #
                # # print("measurement_train mult coeffs ", measurement_train)
                #
                # # print("Train measurements")
                # noisy_train_measurements.append(
                #     self.train_measurements_struc.encode(measurement_train, noisy=True))
                # # print("Test measurements")
                # noisy_test_measurements.append(
                #     self.test_measurements_struc.encode(measurement_test, noisy=True))
                #
                # if self.verbose:
                #     print("i: {}, data_pressure: {} ".format(i, pressure_vec))
                # ref_params['pressure_field'] = pressure_vec
                #
                # iter_state = self.state_struc.encode_state(ref_params)
                # state_data_iters.append(iter_state)
                #
                meas_model_iter_time.append(prec_time)
                meas_model_iter_flux.append(prec_flux)

        # import matplotlib.pyplot as plt
        # train_meas = np.array(noisy_train_measurements)
        # for i in range(train_meas.shape[1]):
        #     plt.scatter(np.cumsum(meas_model_iter_time), train_meas[:, i], label="SoilMoistMin_{}".format(i))
        # plt.legend()
        # plt.show()

        #exit()
        return noisy_train_measurements, noisy_test_measurements, meas_model_iter_time, meas_model_iter_flux

    def run(self):
        #############################
        ### Generate measurements ###
        #############################
        if "measurements_file" in self.measurements_config:
            noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux = load_data(self.train_measurements_struc,
                                                                                             self.test_measurements_struc,
                                                                                             data_csv=self.measurements_config["measurements_file"],
                                                                                             measurements_config=self.measurements_config)
            print("meas_model_iter_flux ", meas_model_iter_flux)

            precipitation_list = []
            for (time_prec, precipitation) in meas_model_iter_flux:
                precipitation_list.extend([precipitation] * time_prec)
            self.measurements_config["precipitation_list"] = precipitation_list

            noisy_measurements, noisy_measurements_to_test, meas_model_iter_time, meas_model_iter_flux = self.process_loaded_measurements(noisy_measurements, noisy_measurements_to_test)

            measurements_time_step = self.measurements_config["measurements_time_step"]
            # Why to call model for real data?
            # MS TODO: why this model.run ?
            # pressure_vec = self.model.make_linear_pressure(self.model_config)
            # ref_params = self.state_struc.compose_ref_dict()
            # ref_params['pressure_field'] = pressure_vec
            # precipitation_flux = 0
            # start_time = 0
            # stop_time = 0
            # new_pressure = self.model_run(precipitation_flux, stop_time, start_time, pressure_vec, ref_params)
            sample_variance = np.var(noisy_measurements, axis=0)
            measurement_noise_covariance = np.diag(sample_variance)
            #@TODO: to simplify, There are no shorter periods of rain/no rain than 'measurements_time_step'
            print("[measurements_time_step] * len(meas_model_iter_flux) ", [measurements_time_step] * len(meas_model_iter_flux))
            print("meas_model_iter_time ", meas_model_iter_time)
            print("meas_model_iter_flux ", meas_model_iter_flux)
            self.results.times_measurements = np.cumsum(meas_model_iter_time)
            print("self.results.times_measurements ", self.results.times_measurements)
        else:
            # import cProfile
            # import pstats
            #
            # pr = cProfile.Profile()
            # pr.enable()

            measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, \
            state_data_iters, meas_model_iter_time, meas_model_iter_flux = self.generate_measurements()

            # Configure ParFlow executable paths if needed
            # os.environ['PARFLOW_HOME'] = '/opt/parflow_install'
            # os.environ['PATH'] += ':/opt/parflow_install/bin'
            #
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('cumtime')
            # ps.print_stats(50)

            residuals = noisy_measurements - measurements
            measurement_noise_covariance = np.cov(residuals, rowvar=False)

            # if "flux_eps" in self.model_config:
            #     self.additional_data_len += 1

            self.results.ref_states = np.array(state_data_iters)
            self.results.train_measuremnts_exact = measurements
            self.results.test_measuremnts_exact = measurements_to_test
            print("meas_model_iter_time ", meas_model_iter_time)
            self.results.times_measurements = np.cumsum(meas_model_iter_time)
            print("self.results.times_measurements ", self.results.times_measurements)

        self.results.precipitation_flux_measurements = meas_model_iter_flux

        #self.results.plot_pressure(self.model, state_data_iters)
        #self.results.plot_saturation(self.model)

        #######################################
        ### Unscented Kalman filter setting ###
        ### - Sigma points
        ### - initital state covariance
        ### - UKF metrices
        ########################################
        ukf = self.set_kalman_filter(measurement_noise_covariance)

        #######################################
        ### Kalman filter run ###
        ### For each measurement (time step) ukf.update() and ukf.predict() are called
        ########################################
        self.run_kalman_filter(ukf, noisy_measurements)

        return self.results

    def model_run(self, flux, stop_time, time_step, pressure, params):
        self.model.run(init_pressure=pressure, precipitation_value=flux,
                  state_params=params, start_time=0, stop_time=stop_time, time_step=time_step)
        new_pressure = self.model.get_data(current_time_step=stop_time, data_name="pressure")
        return new_pressure

    def model_iteration(self, precipitation_flux, pressure, params, model_time_step, model_n_time_steps_per_iter):
        # et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35,
        #                   month=6) / 1000 / 24  # mm/day to m/sec
        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/hour

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
        train_measurements = []
        test_measurements = []
        noisy_train_measurements = []
        noisy_test_measurements = []
        state_data_iters = []

        ###################
        ##   Model runs  ##
        ###################
        # Loop through time steps
        pressure_vec = self.model.make_linear_pressure(self.model_config)
        ref_params = self.state_struc.compose_ref_dict()
        ref_params['pressure_field'] = pressure_vec
        # JB TODO: finish GField.ref to simplify the following lines
        state_vec = self.state_struc.encode_state(ref_params)

        # print("ref params ", ref_params)
        # print("state vec ", state_vec)

        print("precipitation list ", self.measurements_config["precipitation_list"])

        total_time = len(self.measurements_config["precipitation_list"])
        print("total time ", total_time)
        meas_model_iter_time = []
        meas_model_iter_flux = []
        for i in range(0, int(total_time), int(self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"])):
            precipitation_step_start = i
            precipitation_step_end = np.min([i + int(self.measurements_config["model_time_step"] * self.measurements_config["model_n_time_steps_per_iter"]), int(total_time)])
            print("prec start: {}, end: {}".format(precipitation_step_start, precipitation_step_end))
            model_time_step = self.measurements_config["model_time_step"]
            prec_time_flux_per_iter = [(len(list(n_times)), flux) for flux, n_times in groupby(self.measurements_config["precipitation_list"][precipitation_step_start:precipitation_step_end])]

            print("prec_time_flux_per_iter ", prec_time_flux_per_iter)

            for (prec_time, prec_flux) in prec_time_flux_per_iter:
                model_n_time_steps_per_iteration = prec_time / model_time_step
                measurement_train, measurement_test, pressure_vec, sat_vec \
                    = self.model_iteration(prec_flux, pressure_vec, ref_params, model_time_step=model_time_step,
                                           model_n_time_steps_per_iter=model_n_time_steps_per_iteration)
                self.results.ref_saturation.append(sat_vec)

                #print("ref params ", ref_params)

                train_measurements.append(self.train_measurements_struc.encode(measurement_train))
                test_measurements.append(self.test_measurements_struc.encode(measurement_test))

                #print("self.train_measurements_struc.z_positions ", self.train_measurements_struc.z_positions())

                calibration_coeffs_z_positions = self.state_struc.get_calibration_coeffs_z_positions()
                if len(calibration_coeffs_z_positions) > 0:
                    measurement_train = self.train_measurements_struc.mult_calibration_coef(self.train_measurements_struc, measurement_train, ref_params["calibration_coeffs"], np.squeeze(calibration_coeffs_z_positions))

                #print("measurement_train mult coeffs ", measurement_train)

                #print("Train measurements")
                noisy_train_measurements.append(
                    self.train_measurements_struc.encode(measurement_train, noisy=True))
                #print("Test measurements")
                noisy_test_measurements.append(
                    self.test_measurements_struc.encode(measurement_test, noisy=True))

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

        return train_measurements, noisy_train_measurements, test_measurements, noisy_test_measurements,\
               state_data_iters, meas_model_iter_time, meas_model_iter_flux

    def get_measurement(self, current_time_step, measurements_struct):
        measurements_dict = {}
        for measurement_name, measure_obj in measurements_struct.items():
            data_to_measure = self.model.get_data(current_time_step=current_time_step, data_name=measurement_name)
            measurements_dict[measurement_name] = measure_obj.interp @ data_to_measure
        return measurements_dict

    # @staticmethod
    # def add_noise_to_measurements(measurements, level=0.1, distr_type="uniform"):
    #     noisy_measurements = np.zeros(measurements.shape)
    #     for i in range(measurements.shape[1]):
    #         noisy_measurements[:, i] = add_noise(measurements[:, i], noise_level=level, distr_type=distr_type)
    #     return noisy_measurements

    #####################
    ### Kalman filter ###
    #####################
    def state_transition_function(self, state_vec, dt, iter_duration, precipitation_flux):
        print("dt: ", dt, "iter duration time: ", iter_duration)
        pid = os.getpid()
        timestamp = int(time.time())

        if os.environ.get("SCRATCHDIR"):
            scratch_dir = os.environ.get("SCRATCHDIR")
            parflow_working_dir = os.path.join(scratch_dir, "parflow_working_dir_{}_{}".format(pid, timestamp))
        else:
            parflow_working_dir = os.path.join(self.model._workdir, "parflow_working_dir_{}_{}".format(pid, timestamp))
        os.makedirs(parflow_working_dir)
        #print("state vec ", state_vec)
        state = self.state_struc.decode_state(state_vec)
        pressure_data = state["pressure_field"]
        flux_eps_std = None

        et_per_time = 0 #ET0(**dict(zip(model_config['evapotranspiration_params']["names"],
                   #model_config['evapotranspiration_params']["values"]))) / 1000 / 24

        iter_duration = float(iter_duration)

        #print("iter duration ", iter_duration)
        #print("precipitation flux ", precipitation_flux)
        #print("state ", state)

        self.model.run(init_pressure=pressure_data, precipitation_value=precipitation_flux,
                       state_params=state, start_time=0, stop_time=iter_duration,
                       time_step=self.kalman_config["model_time_step"],
                       working_dir=parflow_working_dir)

        state["pressure_field"] = self.model.get_data(current_time_step=iter_duration, data_name="pressure")

        measurements_train = self.get_measurement(current_time_step=iter_duration,
                                                  measurements_struct=self.train_measurements_struc)
        measurements_test = self.get_measurement(current_time_step=iter_duration,
                                                 measurements_struct=self.test_measurements_struc)

        new_state_vec = self.state_struc.encode_state(state)

        if self.lock:
            self.state_measurements[tuple(new_state_vec)] = (measurements_train, measurements_test)

        shutil.rmtree(parflow_working_dir)
        return new_state_vec

    def measurement_function(self, state_vec, measurements_type="train"):
        measurements_train_dict, measurements_test_dict = self.state_measurements[tuple(state_vec)]

        if measurements_type == "train":
            calibration_coeffs_z_positions = self.state_struc.get_calibration_coeffs_z_positions()
            if len(calibration_coeffs_z_positions) > 0:
                measurements_train = self.train_measurements_struc.mult_calibration_coef(self.train_measurements_struc,
                                                                                        measurements_train_dict,
                                                                                        self.state_struc.decode_state(state_vec)["calibration_coeffs"],
                                                                                        np.squeeze(calibration_coeffs_z_positions))

                return self.train_measurements_struc.encode(measurements_train)
            else:
                return self.train_measurements_struc.encode(measurements_train_dict)
        elif measurements_type == "test":
            return self.test_measurements_struc.encode(measurements_test_dict)

    @staticmethod
    def get_sigma_points_obj(sigma_points_params, num_state_params):
        return MerweScaledSigmaPoints(n=num_state_params, sqrt_method=sqrt_func, **sigma_points_params, )

    def add_noise_to_init_state(self, init_state, init_pressure_data):

        for key in self.state_struc.keys():
            if key in init_state:
                if key == "pressure_field":
                    init_state["pressure_field"] = add_noise(np.squeeze(init_pressure_data),
                                                             noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                                                             distr_type=self.kalman_config["noise_distr_type"])
                else:
                    noisy_param_value = self.state_struc[key].transform_from_gauss(add_noise(np.array([self.state_struc[key].transform_to_gauss(init_state[key])]),
                                                             noise_level=self.state_struc[key].std,
                                                             distr_type=self.kalman_config["noise_distr_type"]))
                    init_state[key] = np.squeeze(noisy_param_value)
        return init_state

    def set_kalman_filter(self,  measurement_noise_covariance):
        num_state_params = self.state_struc.size()
        dim_z = measurement_noise_covariance.shape[0]   # Number of measurement inputs

        sigma_points_params = self.kalman_config["sigma_points_params"]

        #sigma_points = JulierSigmaPoints(n=n, kappa=1)
        sigma_points = KalmanFilter.get_sigma_points_obj(sigma_points_params, num_state_params)
        #sigma_points = MerweScaledSigmaPoints(n=num_state_params, alpha=sigma_points_params["alpha"], beta=sigma_points_params["beta"], kappa=sigma_points_params["kappa"], sqrt_method=sqrt_func)

        # Initialize the UKF filter
        time_step = 1 # one hour time step
        # ukf = UnscentedKalmanFilter(dim_x=num_state_params, dim_z=dim_z, dt=time_step,
        #                             fx=self.state_transition_function, #KalmanFilter.state_transition_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
        #                             hx=self.measurement_function, #KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
        #                             points=sigma_points)

        ukf = ParallelUKF(dim_x=num_state_params, dim_z=dim_z,
                               dt=time_step,
                               fx=self.state_transition_function,
                               hx=self.measurement_function,
                               points=sigma_points)

        Q_state = self.state_struc.compose_Q()
        ukf.Q = Q_state
        print("ukf.Q.shape ", ukf.Q.shape)
        print("ukf.Q ", ukf.Q)
        print("diag ukf.Q ", np.diag(ukf.Q))
        ukf.R = measurement_noise_covariance
        print("R measurement_noise_covariance ", measurement_noise_covariance)

        print("self.model ", self.model)

        data_pressure = self.model.make_linear_pressure(self.model_config)

        #data_pressure = self.model.get_data(current_time_step=0, data_name="pressure")
        print("data pressure ", data_pressure)

        el_centers_z = self.model.get_el_centers_z()
        init_mean, init_cov = self.state_struc.compose_init_state(el_centers_z)
        #print("init mean ", init_mean)

        init_state = self.state_struc.decode_state(init_mean)
        init_state = self.add_noise_to_init_state(init_state, data_pressure)
        # JB TODO: use init_mean, implement random choice of ref using init distr
        ukf.x = self.state_struc.encode_state(init_state) #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector

        init_cov_multiplicator = self.kalman_config["init_cov_P_multiplicator"] if "init_cov_P_multiplicator" in self.kalman_config else 1

        # print("init_cov_multiplicator ", init_cov_multiplicator)
        # print("init cov ", init_cov)

        ukf.P = init_cov * init_cov_multiplicator  # Initial state covariance matrix

        print("init cov ", init_cov.shape)
        print("np.diag(init_cov) ", np.diag(init_cov))

        return ukf

    def run_kalman_filter(self, ukf, noisy_measurements):
        iter_durations = [self.results.times_measurements[0]] + list(np.array(self.results.times_measurements[1:]) - np.array(self.results.times_measurements[:-1]))
        print("iter durations ", iter_durations)

        for i, measurement in enumerate(noisy_measurements):
            ukf.predict(iter_duration=iter_durations[i], precipitation_flux=self.results.precipitation_flux_measurements[i])
            ukf.update(measurement)
            print("sum ukf.P ", np.sum(ukf.P))
            #ukf.residual_z
            print("Estimated State:", ukf.x)
            self.results.times.append(self.results.times_measurements[i])
            self.results.ukf_x.append(ukf.x)
            self.results.ukf_P.append(ukf.P)
            self.results.measurement_in.append(measurement)

            measurements_train_dict, measurements_test_dict = self.state_measurements[list(self.state_measurements.keys())[-1]]
            self.results.ukf_train_meas.append(self.train_measurements_struc.encode(measurements_train_dict))
            self.results.ukf_test_meas.append(self.test_measurements_struc.encode(measurements_test_dict))

        joblib.dump(self.results, self.work_dir / 'kalman_results.pkl')

        return self.results


    #
    # def postprocess_data(self, state_data_iters, pred_state_data_iter):
    #     iter_mse_pressure_data = []
    #     iter_mse_train_measurements = []
    #     iter_mse_test_measurements = []
    #
    #     iter_mse_model_config_data = {}
    #
    #     print("len additional data ", self.additional_data_len)
    #
    #     for state_data, pred_state_data  in zip(state_data_iters, pred_state_data_iter):
    #         print("len state data ", len(state_data))
    #         print("len pred state data ", len(pred_state_data))
    #         len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]) + len(
    #             self.model_config["params"]["names"])
    #
    #         if "flux_eps" in self.model_config:
    #             pred_state_data = pred_state_data[:-1]
    #
    #         pressure_data = state_data[:-self.additional_data_len]
    #         pred_pressure_data = pred_state_data[:-self.additional_data_len]
    #         print("len pressure data ", len(pressure_data))
    #
    #         print("pressure data ", pressure_data)
    #         print("pred pressure data ", pred_pressure_data)
    #
    #         print("len pressure data ", len(pressure_data))
    #         print("len pred pressure data ", len(pred_pressure_data))
    #
    #         iter_mse_pressure_data.append(np.linalg.norm(pressure_data - pred_pressure_data))
    #
    #         train_measurements = state_data[-self.additional_data_len: -self.additional_data_len + len(self.kalman_config["mes_locations_train"])]
    #         pred_train_measurements = pred_state_data[-self.additional_data_len: -self.additional_data_len + len(
    #             self.kalman_config["mes_locations_train"])]
    #
    #         iter_mse_train_measurements.append(np.linalg.norm(train_measurements - pred_train_measurements))
    #
    #         if self.additional_data_len == len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]):
    #             test_measurements = state_data[-self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
    #             pred_test_measurements = pred_state_data[ -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
    #         else:
    #             test_measurements = state_data[
    #                                 -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
    #                                 -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
    #                                 + len(self.kalman_config["mes_locations_test"])]
    #
    #             pred_test_measurements = pred_state_data[
    #                                      -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
    #                                      -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
    #                                      + len(self.kalman_config["mes_locations_test"])]
    #
    #         iter_mse_test_measurements.append(np.linalg.norm(test_measurements - pred_test_measurements))
    #
    #         if len(self.model_config["params"]["names"]) > 0:
    #             for idx, param_name in enumerate(self.model_config["params"]["names"]):
    #                 l2_norm = np.linalg.norm(state_data[-len(self.model_config["params"]["names"]) + idx] - pred_state_data[-len(self.model_config["params"]["names"]) + idx])
    #
    #                 iter_mse_model_config_data.setdefault(param_name, []).append(l2_norm)
    #
    #
    #     print("iter_mse_pressure_data ", iter_mse_pressure_data)
    #     print("iter_mse_train_measurements ", iter_mse_train_measurements)
    #     print("iter_mse_test_measurements ", iter_mse_test_measurements)
    #     print("iter_mse_model_config_data ", iter_mse_model_config_data)
    #


#@memory.cache
def run_kalman(workdir, cfg_file):
    kalman_filter = KalmanFilter.from_config(Path(workdir), Path(cfg_file).resolve(), verbose=False)
    return kalman_filter.run()


def main():
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
    #os.environ['PARFLOW_HOME'] = '/opt/parflow_install'
    #os.environ['PATH'] += ':/opt/parflow_install/bin'

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)

