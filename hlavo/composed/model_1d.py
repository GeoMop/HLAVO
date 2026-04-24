from __future__ import annotations

import logging
from pathlib import Path

import attrs
import numpy as np
import yaml
from dask.distributed import Queue

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.ingress.moist_profile.load_data import load_data
from hlavo.kalman.kalman import KalmanFilter

LOG = logging.getLogger(__name__)
TIME_ORIGIN = np.datetime64("2000-01-01T00:00:00", "ms")
MILLISECONDS_PER_DAY = 86_400_000.0


def datetime64_to_model_time_days(date_time: np.datetime64) -> float:
    delta_ms = (np.datetime64(date_time, "ms") - TIME_ORIGIN) / np.timedelta64(1, "ms")
    return float(delta_ms) / MILLISECONDS_PER_DAY


@attrs.define(frozen=True)
class Model1DLocation:
    idx: int
    longitude: float
    latitude: float


KalmanFilterMock = KalmanFilter


class Model1D:
    def __init__(self, idx, initial_state=0.0, work_dir=None, kalman_config=None, location=None):
        self.idx = idx
        self.state = initial_state
        self.location = location
        self.work_dir = Path(work_dir).resolve() if work_dir is not None else None
        self.kalman_config_path = Path(kalman_config).resolve() if kalman_config is not None else None
        self.kalman = None
        self.ukf = None
        self._rng = np.random.default_rng(seed=10_000 + int(idx))

        if kalman_config is not None:
            with self.kalman_config_path.open("r", encoding="utf-8") as handle:
                main_cfg_data = yaml.safe_load(handle) or {}
            self.kalman = KalmanFilterMock.from_config(self.work_dir, main_cfg_data, verbose=False)
            if self._resolve_kalman_measurements_file():
                self.ukf = self.prepare_kalman_measurements()

    def _resolve_kalman_measurements_file(self) -> bool:
        assert self.kalman is not None
        assert self.kalman_config_path is not None

        data_csv_raw = self.kalman.measurements_config.get("measurements_file")
        if data_csv_raw is None:
            LOG.info("[1D %s] measurements_file not provided; skipping Kalman preloaded measurements.", self.idx)
            return False

        data_csv_path = Path(str(data_csv_raw))
        if data_csv_path.is_absolute():
            resolved_path = data_csv_path
        else:
            resolved_path = (self.kalman_config_path.parent / data_csv_path).resolve()

        assert resolved_path.exists(), f"measurements_file does not exist: {resolved_path}"
        self.kalman.measurements_config["measurements_file"] = str(resolved_path)
        LOG.info("[1D %s] resolved measurements_file to %s", self.idx, resolved_path)
        return True

    def prepare_kalman_measurements(self):
        assert self.kalman is not None

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

        (
            noisy_measurements,
            noisy_measurements_to_test,
            measurement_state_flag_sampled,
            meas_model_iter_time,
            meas_model_iter_flux,
        ) = self.kalman.process_loaded_measurements(
            noisy_measurements,
            noisy_measurements_to_test,
            measurement_state_flag,
        )
        _ = measurement_state_flag_sampled

        sample_variance = np.nanvar(noisy_measurements, axis=0)
        measurement_noise_covariance = np.diag(sample_variance)

        self.kalman.results.times_measurements = np.cumsum(meas_model_iter_time)
        self.kalman.results.precipitation_flux_measurements = meas_model_iter_flux

        return self.kalman.set_kalman_filter(measurement_noise_covariance)

    def step(self, date_time, data_for_step):
        LOG.info(
            "[1D %s] step at date_time=%s, data=%s, current_state=%s",
            self.idx,
            date_time,
            data_for_step,
            self.state,
        )
        self.state += data_for_step
        LOG.info("[1D %s] new state=%s", self.idx, self.state)
        return self.state

    def run_loop(self, t_end, queue_name_in, queue_name_out):
        q_in = Queue(queue_name_in)
        q_out = Queue(queue_name_out)

        current_time = 0.0
        while current_time < t_end:
            msg_in = q_in.get()
            assert isinstance(msg_in, Data3DTo1D), f"Unexpected 3D->1D payload: {type(msg_in)}"
            assert msg_in.site_id == self.idx, f"Expected site_id {self.idx}, got {msg_in.site_id}"

            self.step(msg_in.date_time, msg_in.pressure_head)
            contribution = float(self._rng.uniform(5.0e-5, 5.0e-4))
            q_out.put(
                Data1DTo3D(
                    date_time=msg_in.date_time,
                    site_id=self.idx,
                    longitude=float(self.location.longitude),
                    latitude=float(self.location.latitude),
                    velocity=contribution,
                )
            )
            LOG.info("[1D %s] sent contribution=%s at date_time=%s", self.idx, contribution, msg_in.date_time)
            current_time = datetime64_to_model_time_days(msg_in.date_time)

        LOG.info("[1D %s] finished loop at t=%s (t_end=%s)", self.idx, current_time, t_end)
        return f"1D model {self.idx} done; final state={self.state}"


Model1DMock = Model1D


def model1d_worker_entry(idx, t_end, queue_name_in, queue_name_out, work_dir, kalman_config, location):
    model = Model1D(
        idx=idx,
        initial_state=0.0,
        work_dir=work_dir,
        kalman_config=kalman_config,
        location=location,
    )
    return model.run_loop(t_end, queue_name_in, queue_name_out)
