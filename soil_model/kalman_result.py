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
    """
    Transpose the state variable array if not None.

    :param state_var: Input state variable array or None
    :return: Transposed array if not None, otherwise None
    """
    if state_var is None:
        return None
    return np.array(state_var).T


@attrs.define
class KalmanResults:
    """
    Container class for storing and visualizing results from Kalman filtering.

    :param workdir: Directory for saving plots and output data
    :param data_z: Vertical coordinate (e.g., depth or height)
    :param state_struc: Structure describing the encoded state representation
    :param train_measurements_struc: Measurement structure for training data
    :param test_measurements_struc: Measurement structure for test data
    :param cfg: Configuration dictionary (e.g., plotting options)
    """

    workdir: Path
    data_z: np.ndarray
    state_struc: StateStructure
    train_measurements_struc: MeasurementsStructure
    test_measurements_struc: MeasurementsStructure
    cfg: Dict[str, Any]

    times: List[float] = attrs.field(factory=list)
    times_measurements: List[float] = attrs.field(factory=list)
    precipitation_flux_measurements: List[float] = attrs.field(factory=list)
    ref_states: List[np.ndarray] = attrs.field(factory=list)
    ukf_x: List[np.ndarray] = attrs.field(factory=list)
    ukf_P: List[np.ndarray] = attrs.field(factory=list)
    train_measuremnts_exact: List[np.ndarray] = attrs.field(factory=list)
    test_measuremnts_exact: List[np.ndarray] = attrs.field(factory=list)
    measurement_in: List[np.ndarray] = attrs.field(factory=list)
    ref_saturation: List[np.ndarray] = attrs.field(factory=list)
    ukf_train_meas: List[np.ndarray] = attrs.field(factory=list)
    ukf_test_meas: List[np.ndarray] = attrs.field(factory=list)
    moistures: List[np.ndarray] = attrs.field(factory=list)
    velocities: List[np.ndarray] = attrs.field(factory=list)

    def plot_pressure(self, model, state_data_iter):
        """
        Plot decoded pressure fields from a sequence of state vectors.

        :param model: Model instance providing a `plot_pressure()` method
        :param state_data_iter: Iterable of state vectors to decode and plot
        :return: None
        """
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in state_data_iter]
        pressure = np.array(pressure)
        model.plot_pressure(pressure, self.times_measurements)

    def plot_pressure_ref(self):
        """
        Plot reference (true) pressure and saturation fields.

        :return: None
        """
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in self.ref_states]
        pressure = np.array(pressure)
        sat = np.array(self.ref_saturation)
        output = RichardsSolverOutput(self.times, pressure, sat, None, None, self.data_z)
        plot_richards_output(output, [], self.workdir / "ref_solution.pdf")

    def plot_pressure_mean(self):
        """
        Plot mean pressure fields estimated from the UKF.

        :return: None
        """
        pressure = [self.state_struc.decode_state(state_vec)["pressure_field"] for state_vec in self.ukf_x]
        pressure = np.array(pressure)
        sat = pressure  # Placeholder for full VG parameter support
        output = RichardsSolverOutput(self.times, pressure, sat, None, None, self.data_z)
        plot_richards_output(output, [], self.workdir / "mean_solution.pdf")

    def plot_saturation(self, model):
        """
        Plot reference saturation profiles over time.

        :param model: Model instance providing a `plot_saturation()` method
        :return: None
        """
        saturation = np.array(self.ref_saturation)
        print("SATURATION shape", saturation.shape)
        model.plot_saturation(saturation, self.times_measurements)

    def plot_results(self):
        """
        Execute full result visualization:
          - Reference and UKF mean pressure plots
          - Covariance eigenvalue diagnostics
          - Model parameter uncertainties
          - Measurement comparisons (train/test)

        :return: None
        """
        if len(self.ref_states) > 0:
            self.plot_pressure_ref()
        self.plot_pressure_mean()
        self._plot_model_params()

        n_times = len(self.times)
        for i in range(0, n_times, 2):
            covariance_plot(self.ukf_P[i], self.times[i], self.state_struc, n_evec=5, show=False)

        self._plot_measurements(meas_type="train")
        self._plot_measurements(meas_type="test")

    def _plot_heatmap(self, cov_matrix):
        """
        Plot correlation heatmap derived from a covariance matrix.

        :param cov_matrix: Covariance matrix to analyze
        :return: None
        """
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        eigenvalues = np.linalg.eigvals(cov_matrix)

        if np.any(eigenvalues < 0):
            print("Warning: Covariance matrix is not positive semi-definite!")

        std_devs = np.sqrt(np.diag(cov_matrix))
        diag_matrix = np.diag(std_devs)
        correlation_matrix = np.linalg.inv(diag_matrix) @ cov_matrix @ np.linalg.inv(diag_matrix)
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix = np.clip(correlation_matrix, -1, 1)

        sns.heatmap(correlation_matrix, cbar=True, cmap='coolwarm', annot=False, ax=axes)
        axes.set_title('Correlation Matrix Heatmap')
        axes.set_xlabel('Variables')
        axes.set_ylabel('Variables')
        fig.savefig("heatmap.pdf")
        plt.show()

    def _decode_meas(self, state_array_list):
        """
        Decode measurement-related variables from state arrays.

        :param state_array_list: List of encoded state vectors
        :return: Dictionary of decoded measurement arrays or None
        """
        state_array = trans_state(state_array_list)
        if state_array is None:
            return None
        states_dict = self.state_struc.decode_state(state_array)
        param_dict = {k: v for k, v in states_dict.items() if isinstance(self.state_struc[k], Measure)}
        return param_dict

    def _plot_measurements(self, meas_type="train"):
        """
        Plot predicted, exact, and noisy measurements over time.

        :param meas_type: Type of dataset ("train" or "test")
        :return: None
        """
        times = np.array(self.times)
        measurements_struc = self.train_measurements_struc if meas_type == "train" else self.test_measurements_struc

        import matplotlib
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        matplotlib.rcParams.update({'font.size': 13})

        if meas_type == 'train':
            meas_in_all = measurements_struc.decode(np.array(self.measurement_in).T)
        else:
            meas_in_all = None

        meas_exact_all = self.train_measuremnts_exact if meas_type == "train" else self.test_measuremnts_exact
        ukf_meas_all = self.ukf_train_meas if meas_type == "train" else self.ukf_test_meas

        meas_exact_dict = {}
        if len(meas_exact_all) > 0:
            meas_exact_dict = measurements_struc.decode(meas_exact_all.T)

        meas_x_all = measurements_struc.decode(np.array(ukf_meas_all).T)

        for measurement_name, measure_obj in measurements_struc.items():
            meas_x = meas_x_all[measurement_name]
            meas_exact = meas_exact_dict.get(measurement_name, None)
            n_meas = len(meas_x)
            fig, ax = plt.subplots(figsize=(20, 10))
            meas_z = measure_obj.z_pos
            colors = sns.color_palette("tab10")

            for i in range(n_meas):
                col = colors[i % 10]
                if meas_exact is not None:
                    ax.scatter(times, meas_exact[i], c=col, s=30, marker='o', label=f"obs(z={meas_z[i]})")
                if meas_in_all is not None and measurement_name in meas_in_all:
                    meas_in = meas_in_all[measurement_name]
                    ax.scatter(times, meas_in[i], c=col, s=30, marker='x', label=f"noisy obs(z={meas_z[i]})")
                ax.plot(times, meas_x[i], c=col, linestyle='--', linewidth=2, label=f"pred(z={meas_z[i]})")

                ax.set_xlabel("time [min]")
                ax.set_ylabel(f"{meas_type} {measurement_name}")
            fig.legend()
            fig.tight_layout()
            fig.savefig(self.workdir / f"{meas_type}_{measurement_name}_loc.pdf")
            if self.cfg['show']:
                plt.show()

    def _decode_params(self, state_array_list):
        """
        Decode model parameters and calibration coefficients.

        :param state_array_list: List or array of encoded state vectors
        :return: Dictionary of decoded parameters
        """
        state_array = np.array(state_array_list).T
        states_dict = self.state_struc.decode_state(state_array)
        param_dict = {k: v for k, v in states_dict.items() if isinstance(self.state_struc[k], (GVar, CalibrationCoeffs))}
        return param_dict

    def plot_calibration_coeffs(self, ref_values, values, vars):
        """
        Plot evolution of calibration coefficients and uncertainty bounds.

        :param ref_values: Reference calibration coefficient values
        :param values: Estimated coefficient means
        :param vars: Estimated coefficient variances
        :return: None
        """
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

            # Asymmetric error bars
            lower_err = median_values - q05_values
            upper_err = q95_values - median_values

            ax.plot(self.times, calib_ref_value, label=f"exact pos: {z_position}")
            ax.errorbar(self.times, median_values, yerr=[lower_err, upper_err], fmt='o', capsize=5,
                        label=f"calibration_coeff, pos: {z_position}")
            ax.set_ylabel(f"pos: {z_position}")

        fig.legend()
        fig.savefig(self.workdir / "calibration_coeffs.pdf")
        if self.cfg['show']:
            plt.show()


    def _plot_model_params_quantiles(self, ref_params, x_params, var_params):
        """
        Plot model parameter quantiles (5%, 50%, 95%) with uncertainty bounds.

        :param ref_params: Reference (true) parameter values
        :param x_params: Estimated parameter mean values
        :param var_params: Estimated parameter variances
        :return: None
        """
        median_values = {}
        q05_values = {}
        q95_values = {}

        for var_key, values in x_params.items():
            encoded_values = self.state_struc[var_key].encode(values)
            encoded_vars = self.state_struc[var_key].encode(var_params[var_key])

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
                if k in ref_params:
                    ax.plot(self.times, ref_params[k], label=f"{k}_exact")

                lower_err = median_values[k] - q05_values[k]
                upper_err = q95_values[k] - median_values[k]

                ax.errorbar(self.times, median_values[k], yerr=[lower_err, upper_err],
                            fmt='o', capsize=5, label=f"{k}_kalman")
                ax.set_ylabel(k)

            fig.legend()
            fig.savefig(self.workdir / f"model_param_{k}_q_err.pdf")
            if self.cfg['show']:
                plt.show()

    def _plot_model_params(self):
        """
        Decode and visualize model parameter evolution over time.

        This includes:
          - Decoding UKF-estimated parameters and variances
          - Comparing with reference parameters (if available)
          - Plotting calibration coefficient uncertainties separately

        :return: None
        """
        import matplotlib
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        matplotlib.rcParams.update({'font.size': 13})

        x_params = self._decode_params(self.ukf_x)
        P_diag = np.diagonal(self.ukf_P, axis1=1, axis2=2)
        var_params = self._decode_params(P_diag)

        ref_params = {}
        if len(self.ref_states):
            ref_params = self._decode_params(self.ref_states)
            if "calibration_coeffs" in ref_params:
                self.plot_calibration_coeffs(
                    ref_params["calibration_coeffs"],
                    x_params["calibration_coeffs"],
                    var_params["calibration_coeffs"],
                )
                del ref_params["calibration_coeffs"]
                del x_params["calibration_coeffs"]
                del var_params["calibration_coeffs"]

        self._plot_model_params_quantiles(ref_params, x_params, var_params)

    def postprocess(self):
        """
        Perform full postprocessing after UKF estimation.

        Steps include:
          - Generating pressure, covariance, and measurement plots
          - Plotting parameter uncertainty evolution
          - Creating correlation heatmaps of the final covariance matrix

        :return: None
        """
        self.plot_results()
        self._plot_heatmap(cov_matrix=self.ukf_P[-1])

    def generate_measurement_plot(self):
        """
        Generate comparison plots for measured and noisy measurement data.

        Note:
            This method assumes that `measurements`, `noisy_measurements`,
            and `data_name` variables exist in the current scope or class.

        :return: None
        """
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

