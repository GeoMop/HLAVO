from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from dask.distributed import Queue

from hlavo.composed.composed_protocol import Data1DTo3D, Data3DTo1D
from hlavo.ingress.moist_profile.load_data import load_data
from hlavo.kalman.kalman import KalmanFilter

LOGGER = logging.getLogger(__name__)


class Model1D:
    def __init__(
        self,
        idx: int,
        initial_state: float = 0.0,
        work_dir: Path | None = None,
        kalman_config: Path | None = None,
    ):
        if kalman_config is None:
            raise ValueError("kalman_config must be provided")
        self.idx = idx
        self.state = float(initial_state)
        self.kalman = KalmanFilter.from_config(work_dir, kalman_config, verbose=False)
        self.ukf = self.prepare_kalman_measurements()

    def prepare_kalman_measurements(self):
        noisy_measurements, noisy_measurements_to_test, meas_model_iter_flux, measurement_state_flag = load_data(
            self.kalman.train_measurements_struc,
            self.kalman.test_measurements_struc,
            data_csv=self.kalman.measurements_config["measurements_file"],
            measurements_config=self.kalman.measurements_config,
        )

        precipitation_list = []
        for time_prec, precipitation in meas_model_iter_flux:
            precipitation_list.extend([precipitation] * time_prec)
        self.kalman.measurements_config["precipitation_list"] = precipitation_list

        noisy_measurements, _noisy_measurements_to_test, _measurement_state_flag_sampled, meas_model_iter_time, meas_model_iter_flux = (
            self.kalman.process_loaded_measurements(
                noisy_measurements,
                noisy_measurements_to_test,
                measurement_state_flag,
            )
        )

        sample_variance = np.nanvar(noisy_measurements, axis=0)
        measurement_noise_covariance = np.diag(sample_variance)
        self.kalman.results.times_measurements = np.cumsum(meas_model_iter_time)
        self.kalman.results.precipitation_flux_measurements = meas_model_iter_flux
        return self.kalman.set_kalman_filter(measurement_noise_covariance)

    def step(self, data_for_step: Data3DTo1D) -> Data1DTo3D:
        LOGGER.info(
            "[1D %s] step at t=%s, pressure_head=%s, current_state=%s",
            self.idx,
            data_for_step.date_time,
            data_for_step.pressure_head,
            self.state,
        )
        self.state += data_for_step.pressure_head
        LOGGER.info("[1D %s] new state=%s", self.idx, self.state)
        return Data1DTo3D.build(
            date_time=data_for_step.date_time,
            site_id=self.idx,
            velocity=self.state,
        )

    def run_loop(self, t_end: np.datetime64, queue_name_in: str, queue_name_out: str):
        q_in = Queue(queue_name_in)
        q_out = Queue(queue_name_out)
        current_time = None

        while True:
            payload = q_in.get()
            if payload is None:
                break
            if not isinstance(payload, Data3DTo1D):
                raise TypeError(f"Expected Data3DTo1D on queue, got {type(payload)}")
            if payload.site_id != self.idx:
                raise ValueError(f"Received payload for site_id={payload.site_id}, expected {self.idx}")

            contribution = self.step(payload)
            q_out.put(contribution)
            current_time = payload.date_time
            LOGGER.info(
                "[1D %s] sent contribution=%s at t=%s",
                self.idx,
                contribution.velocity,
                payload.date_time,
            )

        LOGGER.info("[1D %s] finished loop at t=%s (t_end=%s)", self.idx, current_time, t_end)
        return f"1D model {self.idx} done; final state={self.state}"


def model1d_worker_entry(
    idx: int,
    t_end: np.datetime64,
    queue_name_in: str,
    queue_name_out: str,
    work_dir: Path,
    kalman_config: Path,
):
    model = Model1D(idx=idx, initial_state=0.0, work_dir=work_dir, kalman_config=kalman_config)
    return model.run_loop(t_end, queue_name_in, queue_name_out)

