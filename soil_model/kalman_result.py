import attrs
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from kalman_state import StateStructure, GVar, Measure, MeasurementsStructure, CalibrationCoeffs
from plots import plot_richards_output, RichardsSolverOutput, covariance_plot


def trans_state(state_var):
    if state_var is None:
        return None
    return np.array(state_var).T


@attrs.define
class KalmanResults:
    workdir: Path
    data_z: np.ndarray
    state_struc: StateStructure
    train_measurements_struc: MeasurementsStructure
    test_measurements_struc: MeasurementsStructure
    cfg: Dict[str, Any]

    times: List[float] = attrs.field(factory=list)  # List of times
    times_measurements: List[float] = attrs.field(factory=list)
    precipitation_flux_measurements: List[float] = attrs.field(factory=list)
    ref_states: List[np.ndarray] = attrs.field(factory=list)  # List of states
    ukf_x: List[np.ndarray] = attrs.field(factory=list)  # List of states
    ukf_P: List[np.ndarray] = attrs.field(factory=list)  # List of states
    train_measuremnts_exact: List[np.ndarray] = attrs.field(factory=list)
    test_measuremnts_exact: List[np.ndarray] = attrs.field(factory=list)
    measurement_in: List[np.ndarray] = attrs.field(factory=list)  # List of states
    ref_saturation: List[np.ndarray] = attrs.field(factory=list)  # List of states
    #mean_saturation: List[np.ndarray]  = attrs.field(factory=list)  # List of states
    ukf_train_meas: List[np.ndarray]  = attrs.field(factory=list)  # List of states
    ukf_test_meas: List[np.ndarray]  = attrs.field(factory=list)  # List of states

    # Future, measurements using StateStruc or similar structure
    #measurements: List[np.ndarray] = attrs.field(default=list)  # List of measurements
    def plot_pressure(self, model, state_data_iter):
        # for state_vec in state_data_iter:
        #     print("state vec ", state_vec)
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in state_data_iter]
        pressure = np.array(pressure)
        #print("PRESSURE shape", pressure.shape)
        model.plot_pressure(pressure, self.times_measurements)

    def plot_pressure_ref(self):
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in self.ref_states]
        pressure = np.array(pressure)
        sat = np.array(self.ref_saturation)
        output = RichardsSolverOutput(self.times, pressure, sat, None, None, self.data_z)
        plot_richards_output(output, [], self.workdir / "ref_solution.pdf")

    def plot_pressure_mean(self):
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in self.ukf_x]
        pressure = np.array(pressure)
        sat = pressure # not available yet, need to improve API of the model to support spatial vG parameters, etc.
        output = RichardsSolverOutput(self.times, pressure, sat, None, None, self.data_z)
        plot_richards_output(output, [], self.workdir / "mean_solution.pdf")

    def plot_saturation(self, model):
        print(len(self.ref_saturation))
        #pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for saturation in self.ref_saturation]
        saturation = np.array(self.ref_saturation)
        print("SATURATION shape", saturation.shape)
        model.plot_saturation(saturation, self.times_measurements)

    def plot_results(self):
        #print("state_loc_measurements ", pred_loc_measurements)
        #print("noisy_measurements ", noisy_measurements)

        #print("ukf_p_var_iter.shape ", ukf_p_var_iter.shape)
        #print("pred model params shape ", np.array(pred_model_params).shape)

        model_params_variances = None
        # pred_loc_measurements_variances = []
        # test_pred_loc_measurements_variances = []
        # if ukf_p_var_iter is not None:
        #     if len(pred_model_params)> 0:
        #         model_params_variances = ukf_p_var_iter[:, -len(pred_model_params[0]):]
        #         print("model params variances ", model_params_variances)
        #
        #     pred_loc_measurements_variances = ukf_p_var_iter[:, -self.additional_data_len: -self.additional_data_len + len(
        #         self.kalman_config["mes_locations_train"])]
        #
        #     if self.additional_data_len == len(self.kalman_config["mes_locations_train"]) + len(
        #             self.kalman_config["mes_locations_test"]):
        #         test_pred_loc_measurements_variances = ukf_p_var_iter[:,
        #                             -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
        #     else:
        #         test_pred_loc_measurements_variances = ukf_p_var_iter[:,
        #                             -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
        #                             -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
        #                             + len(self.kalman_config["mes_locations_test"])]
        #
        #     print("pred_loc_measurements_variances shape ", pred_loc_measurements_variances.shape)
        #     print("test_pred_loc_measurements_variances shape ", test_pred_loc_measurements_variances.shape)
        #

        #times = np.arange(1, pred_loc_measurements.shape[0] + 1, 1)

        #print("times.shape ", times.shape)
        #print("xs[:, 0].shape ", pred_loc_measurements[:, 0].shape)

        #np.save(self.work_dir / "times", times)
        #np.save(self.work_dir / "model_params_variances", model_params_variances)
        #np.save(self.work_dir / "pred_loc_measurements_variances", pred_loc_measurements_variances)
        #np.save(self.work_dir / "test_pred_loc_measurements_variances", test_pred_loc_measurements_variances)

        # plt.scatter(times, pred_loc_measurements[:, 0], marker="o", label="predictions")
        # plt.scatter(times, measurements[:, 0], marker='x',  label="measurements")
        # plt.scatter(times, noisy_measurements[:, 0], marker='x',  label="noisy measurements")
        # plt.legend()
        # plt.show()

        #######
        # Plot model params data
        ######
        #print("pred model params ", pred_model_params)
        #print("pred_model_params shape ", np.array(pred_model_params).shape)
        self.plot_pressure_ref()
        self.plot_pressure_mean()
        self._plot_model_params()

        n_times = len(self.times)
        for i in range(0, n_times, 2):
            covariance_plot(self.ukf_P[i], self.times[i], self.state_struc, n_evec=5, show=False)

        self._plot_measurements(meas_type="train")
        self._plot_measurements(meas_type="test")

    def _plot_heatmap(self, cov_matrix):
        # Generate a heatmap using seaborn
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

        # print("cov matrix ", cov_matrix)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        # print("eigenvalues ", eigenvalues)

        if np.any(eigenvalues < 0):
            print("Warning: Covariance matrix is not positive semi-definite!")

        # print("np.diag(cov_matrix) ", np.diag(cov_matrix))

        # Step 1: Get the standard deviations from the diagonal of the covariance matrix
        std_devs = np.sqrt(np.diag(cov_matrix))

        diag_matrix = np.diag(std_devs)

        # np.linalg.inv(diagonal_matrix) @ cov_matrix @ np.linalg.inv(diagonal_matrix)

        # print("std devs ", std_devs)

        # Step 2: Create a correlation matrix by normalizing the covariance matrix
        # correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

        correlation_matrix = np.linalg.inv(diag_matrix) @ cov_matrix @ np.linalg.inv(diag_matrix)

        # print("np.outer(std_devs, std_devs) ", np.outer(std_devs, std_devs))

        # off_diagonal_elements = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]

        # Find the maximum off-diagonal value
        # max_off_diagonal = np.max(np.abs(off_diagonal_elements))

        np.fill_diagonal(correlation_matrix, 0)

        # Get the indices where values are greater than 1
        indices = np.argwhere(correlation_matrix > 1)

        print("\nIndices where values are greater than 1:")
        for idx in indices:
            print(f"Row {idx[0]}, Column {idx[1]}: Value = {correlation_matrix[idx[0], idx[1]]}")

        correlation_matrix = np.clip(correlation_matrix, -1, 1)

        # sns.heatmap(cov_matrix, cbar=True, cmap='coolwarm', annot=False, ax=axes)
        # # Add title and labels
        # axes.set_title('cov_matrix Matrix Heatmap')
        # axes.set_xlabel('Variables')
        # axes.set_ylabel('Variables')
        # fig.savefig("heatmap.pdf")
        # plt.show()
        # print("correlation matrix ", correlation_matrix)

        sns.heatmap(correlation_matrix, cbar=True, cmap='coolwarm', annot=False, ax=axes)

        # Add title and labels
        axes.set_title('Correlation Matrix Heatmap')
        axes.set_xlabel('Variables')
        axes.set_ylabel('Variables')

        fig.savefig("heatmap.pdf")
        plt.show()

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        #
        # sns.clustermap(cov_matrix, cmap='coolwarm')
        #
        # # Add title and labels
        # axes.set_title('Covariance Matrix Heatmap')
        # axes.set_xlabel('Variables')
        # axes.set_ylabel('Variables')
        #
        # fig.savefig("clustermap.pdf")
        # plt.show()

    def _decode_meas(self, state_array_list):
        state_array = trans_state(state_array_list)
        if state_array is None:
            return None
        states_dict = self.state_struc.decode_state(state_array)
        param_dict = {k: v for k, v in states_dict.items() if isinstance(self.state_struc[k], Measure)}
        return param_dict

    def _plot_measurements(self, meas_type="train"):
        times = np.array(self.times)

        if meas_type == "train":
            measurements_struc = self.train_measurements_struc
        else:
            measurements_struc = self.test_measurements_struc

        import matplotlib
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        matplotlib.rcParams.update({'font.size': 13})

        # #print("n measurements ", n_measurements)
        if meas_type == 'train':
            meas_in_all = measurements_struc.decode(np.array(self.measurement_in).T)
        else:
            meas_in_all = None

        meas_exact_all = self.train_measuremnts_exact if meas_type == "train" else self.test_measuremnts_exact
        ukf_meas_all = self.ukf_train_meas if meas_type == "train" else self.ukf_test_meas

        meas_exact_dict = measurements_struc.decode(meas_exact_all.T)
        #meas_exact_dict = {k: v for k, v in states_dict.items()}

        meas_x_all = measurements_struc.decode(np.array(ukf_meas_all).T)

        measurements_dict = {}
        for measurement_name, measure_obj in measurements_struc.items():
            meas_x = meas_x_all[measurement_name] #measurements_struc.decode(self.ukf_train_meas)   #self._decode_meas(self.ukf_x)[meas_key]
            meas_exact = meas_exact_dict[measurement_name]
            #P_diag = np.diagonal(self.ukf_P, axis1=1, axis2=2)
            #meas_var = self._decode_meas(P_diag)[meas_key]
            #meas_std = np.sqrt(meas_var)
            n_meas = len(meas_x)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            meas_z = measure_obj.z_pos

            colors = sns.color_palette("tab10")
            for i in range(n_meas):
                col = colors[i % 10]
                mse_vs_exact = np.sqrt(np.mean((meas_exact[i] - meas_x[i]) ** 2))
                mse_vs_obs = np.mean((meas_exact[i] - meas_x[i]) ** 2)
                #mean_std = np.mean(meas_std[i])
                #print(f"d=(exact - ukf_est) {i}: std(d) {mse_vs_exact}, std(d)/mean(std_est): {mse_vs_exact / mean_std}")
                #print(f"d=(obs - ukf_est) {i}: std(d) {mse_vs_obs} std(d)/mean(std_est): {mse_vs_exact / mean_std}")

                #print("np.sqrt(pred_loc_measurements_variances[:, i]) ",
                #fig, axes = plt.subplots(1, 1)
                #axes.scatter(times, pred_loc_measurements[:, i], marker="o", label="predictions")
                #ax.errorbar(times, meas_x[i], c=col, ms=5, yerr=meas_std[i], fmt='o', capsize=5, label=f'obs_est(z={meas_z[i]})')
                if meas_exact is not None:
                    ax.scatter(times, meas_exact[i], c=col, s=30, marker='o', label=f"obs(z={meas_z[i]})")

                #ax.plot(times, meas_exact[i], c=col, linewidth=2, label=f"obs_sim(z={meas_z[i]})")
                if meas_in_all is not None and measurement_name in meas_in_all:
                    meas_in = meas_in_all[measurement_name]
                    ax.scatter(times, meas_in[i], c=col, s=30, marker='x', label=f"noisy obs(z={meas_z[i]})")

                ax.plot(times, meas_x[i], c=col, linestyle='--', linewidth=2, label=f"obs_sim(z={meas_z[i]})")
                ax.set_xlabel("time[h]")
                ax.set_ylabel(f"{meas_type} {measurement_name}")
            fig.legend()
            fig.tight_layout()
            fig.savefig(self.workdir / f"{meas_type}_{measurement_name}_loc.pdf")
            if self.cfg['show']:
                plt.show()

    def _decode_params(self, state_array_list):
        state_array = np.array(state_array_list).T
        states_dict = self.state_struc.decode_state(state_array)
        param_dict = {k:v for k, v in states_dict.items() if isinstance(self.state_struc[k], (GVar, CalibrationCoeffs))}
        return param_dict

    def plot_calibration_coeffs(self, ref_values, values, vars):
        calibration_coeffs_z_positions = np.array(self.state_struc.get_calibration_coeffs_z_positions())
        fig, axes = plt.subplots(nrows=len(calibration_coeffs_z_positions), ncols=1, figsize=(10, 5))

        for ax, z_position in zip(axes, calibration_coeffs_z_positions):
            index = np.squeeze(np.where(calibration_coeffs_z_positions == z_position)[0])
            callibration_mult_coeff = values[index]
            calib_ref_value = ref_values[index]
            calib_var = vars[index]

            GVar_instance = self.state_struc.get("calibration_coeffs").GVar_coeffs[index]

            encoded_values = GVar_instance.encode(callibration_mult_coeff)
            encoded_vars = GVar_instance.encode(calib_var)

            median = norm.ppf(0.5, loc=encoded_values, scale=encoded_vars)
            q05 = norm.ppf(0.05, loc=encoded_values, scale=encoded_vars)
            q95 = norm.ppf(0.95, loc=encoded_values, scale=encoded_vars)

            median_values = GVar_instance.decode(median)
            q05_values = GVar_instance.decode(q05)
            q95_values = GVar_instance.decode(q95)

            # print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
            ax.plot(self.times, calib_ref_value, label="exact_pos: {}".format(z_position))
            # ax.hlines(y=mean_value_std[0], xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
            # axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
            # print("variances[:, idx] ", variances[:, idx])
            # Compute asymmetric error bars
            lower_err = median_values - q05_values
            upper_err = q95_values - median_values

            ax.errorbar(self.times, median_values, yerr=[lower_err, upper_err], fmt='o', capsize=5,
                        label="calibration_coeff, pos: {}".format(z_position))  # label='Data with variance')

            # axes.set_xlabel("param_name")
            ax.set_ylabel("pos: {}".format(z_position))
        fig.legend()
        fig.savefig(self.workdir / f"calibration_coeffs.pdf")
        if self.cfg['show']:
            plt.show()


    def _plot_model_params_quantiles(self, ref_params, x_params, var_params):
        median_values = {}
        q05_values = {}
        q95_values = {}

        for var_key, values in x_params.items():
            encoded_values = self.state_struc[var_key].encode(values)
            encoded_vars = self.state_struc[var_key].encode(var_params[var_key])

            #q05, median, q95 = norm.ppf([0.05, 0.5, 0.95], loc=encoded_values, scale=encoded_vars)
            median = norm.ppf(0.5, loc=encoded_values, scale=encoded_vars)
            q05 = norm.ppf(0.05, loc=encoded_values, scale=encoded_vars)
            q95 = norm.ppf(0.95, loc=encoded_values, scale=encoded_vars)

            median_values[var_key] = self.state_struc[var_key].decode(median)
            q05_values[var_key] = self.state_struc[var_key].decode(q05)
            q95_values[var_key] = self.state_struc[var_key].decode(q95)

        n_params = len(x_params)

        if n_params > 0:
            if n_params == 1:
                n_params += 1
            fig, axes = plt.subplots(nrows=n_params, ncols=1, figsize=(10, 5))

            for ax, k in zip(axes, x_params.keys()):
                #print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
                ax.plot(self.times, ref_params[k], label=f"{k}_exact")
                #ax.hlines(y=mean_value_std[0], xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
                #axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
                #print("variances[:, idx] ", variances[:, idx])
                # Compute asymmetric error bars
                lower_err = median_values[k] - q05_values[k]
                upper_err = q95_values[k] - median_values[k]

                ax.errorbar(self.times, median_values[k], yerr=[lower_err, upper_err], fmt='o', capsize=5, label=f"{k}_kalman")# label='Data with variance')

                #axes.set_xlabel("param_name")
                ax.set_ylabel(k)
            fig.legend()
            fig.savefig(self.workdir / f"model_param_{k}_q_err.pdf")
            if self.cfg['show']:
                plt.show()

    def _plot_model_params(self):
        import matplotlib
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        matplotlib.rcParams.update({'font.size': 13})


        ref_params = self._decode_params(self.ref_states)
        x_params = self._decode_params(self.ukf_x)
        P_diag = np.diagonal(self.ukf_P, axis1=1, axis2=2)
        var_params = self._decode_params(P_diag)

        if "calibration_coeffs" in ref_params:
            self.plot_calibration_coeffs(ref_params["calibration_coeffs"], x_params["calibration_coeffs"], var_params["calibration_coeffs"])
            del ref_params["calibration_coeffs"]
            del x_params["calibration_coeffs"]
            del var_params["calibration_coeffs"]



        self._plot_model_params_quantiles(ref_params, x_params, var_params)

        #model_dynamic_params = KalmanFilter.get_nonzero_std_params(model_config["params"])
        #print("model dynamic params ", model_dynamic_params)

        # n_params = len(x_params)
        #
        # if n_params > 0:
        #     if n_params == 1:
        #         n_params += 1
        #     fig, axes = plt.subplots(nrows=n_params, ncols=1, figsize=(10, 5))
        #
        #     for ax, k in zip(axes, x_params.keys()):
        #         #print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
        #         ax.plot(self.times, ref_params[k], label=f"{k}_exact")
        #         #ax.hlines(y=mean_value_std[0], xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
        #         #axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
        #         #print("variances[:, idx] ", variances[:, idx])
        #         ax.errorbar(self.times, x_params[k], yerr=np.sqrt(var_params[k]), fmt='o', capsize=5, label=f"{k}_kalman")# label='Data with variance')
        #
        #         #axes.set_xlabel("param_name")
        #         ax.set_ylabel(k)
        #     fig.legend()
        #     fig.savefig(self.workdir / f"model_param_{k}.pdf")
        #     if self.cfg['show']:
        #         plt.show()



    def postprocess(self):
        ##############################
        ### Results postprocessing ###
        ##############################
        # KalmanFilter.serialize_kalman_filter(ukf, self.work_dir / "kalman_filter.pkl")
        # Serialize the Kalman filter object
        # auxiliary_data = {"additional_data_len": self.additional_data_len}
        #
        # with (self.work_dir / "auxiliary_data.json").open('w') as f:
        #     json.dump(auxiliary_data, f)
        # with (self.work_dir / "model_config.json").open('w') as f:
        #     json.dump(self.model_config, f)
        # with (self.work_dir / "kalman_config.json").open('w') as f:
        #     json.dump(self.kalman_config, f)
        #
        # np.save(self.work_dir / "noisy_measurements", noisy_measurements)
        # np.save(self.work_dir / "pred_loc_measurements", pred_loc_measurements)
        # np.save(self.work_dir / "pred_model_params", pred_model_params)
        # np.save(self.work_dir / "noisy_measurements_to_test", noisy_measurements_to_test)
        # np.save(self.work_dir / "test_pred_loc_measurements", test_pred_loc_measurements)
        # np.save(self.work_dir / "pred_state_data_iter", pred_state_data_iter)
        # np.save(self.work_dir / "ukf_p_var_iter", ukf_p_var_iter)
        # np.save(self.work_dir / "ukf_last_P", ukf_last_P)

        self.plot_results()

        self._plot_heatmap(cov_matrix=self.ukf_P[-1])

        # self.postprocess_data(state_data_iters, pred_state_data_iter)


    def generate_measurement_plot(self):
        times = np.arange(1, len(measurements) + 1, 1)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.scatter(times, measurements[:, 0], marker="o", label="measurements")
        axes.scatter(times, noisy_measurements[:, 0], marker='x', label="noisy measurements")
        axes.set_xlabel("time")
        axes.set_ylabel(data_name)
        fig.savefig("L2_coarse_L1_fine_samples.pdf")
        fig.legend()
        plt.show()

        if measurements.shape[1] > 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.scatter(times, measurements[:, 1], marker="o", label="measurements")
            axes.scatter(times, noisy_measurements[:, 1], marker='x', label="noisy measurements")
            axes.set_xlabel("time")
            axes.set_ylabel(data_name)
            fig.legend()
            plt.show()
