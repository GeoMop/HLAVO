from __future__ import annotations

import logging
from pathlib import Path

import attrs
import numpy as np
from dask.distributed import Queue

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.ingress.moist_profile.load_data import load_data
from hlavo.kalman.kalman import KalmanFilter
from hlavo.misc.class_resolve import resolve_named_class
from hlavo.misc.config import load_config

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


@attrs.define
class KalmanMock:
    fixed_velocity: float = 0.1

    @classmethod
    def from_config(cls, workdir, config_source, verbose=False):
        config_data, _ = load_config(config_source)
        model_1d_cfg = config_data.get("model_1d", {})
        assert isinstance(model_1d_cfg, dict), "model_1d config must be a mapping"
        _ = workdir
        _ = verbose
        return cls(fixed_velocity=float(model_1d_cfg.get("mock_velocity", 0.0)))

    def step(self, date_time, pressure_head) -> float:
        LOG.info(
            "[1D mock] step at date_time=%s, pressure_head=%s -> velocity=%s",
            date_time,
            pressure_head,
            self.fixed_velocity,
        )
        return self.fixed_velocity


Kalman = KalmanFilter


class Model1D:
    def __init__(self, idx, work_dir=None, config_path=None, location=None):
        self.idx = idx
        self.location = location
        self.work_dir = Path(work_dir).resolve()
        self.kalman = None

        resolved_config_path = Path(config_path).resolve()
        main_cfg_data, _ = load_config(resolved_config_path)
        model_1d_cfg = main_cfg_data.get("model_1d", {})
        assert isinstance(model_1d_cfg, dict), "model_1d config must be a mapping"

        kalman_class = self._resolve_kalman_class(model_1d_cfg)
        self.kalman = kalman_class.from_config(self.work_dir, main_cfg_data, verbose=False)
        self._prepare_kalman_runtime(config_dir=resolved_config_path.parent)

    def _resolve_kalman_class(self, model_1d_cfg: dict) -> type:
        kalman_class_name = str(model_1d_cfg.get("kalman_class_name", "KalmanFilter"))
        return resolve_named_class(kalman_class_name, ("hlavo.composed.model_1d", "hlavo.kalman"))

    def _prepare_kalman_runtime(self, config_dir: Path) -> None:
        if not hasattr(self.kalman, "measurements_config"):
            return
        if not hasattr(self.kalman, "set_kalman_filter"):
            return
        if not self._resolve_kalman_measurements_file(config_dir):
            return
        self.kalman.ukf = self.prepare_kalman_measurements()

    def _resolve_kalman_measurements_file(self, config_dir: Path) -> bool:
        assert hasattr(self.kalman, "measurements_config")

        data_csv_raw = self.kalman.measurements_config.get("measurements_file")
        if data_csv_raw is None:
            LOG.info("[1D %s] measurements_file not provided; skipping Kalman preloaded measurements.", self.idx)
            return False

        data_csv_path = Path(str(data_csv_raw))
        if data_csv_path.is_absolute():
            resolved_path = data_csv_path
        else:
            resolved_path = (config_dir / data_csv_path).resolve()

        assert resolved_path.exists(), f"measurements_file does not exist: {resolved_path}"
        self.kalman.measurements_config["measurements_file"] = str(resolved_path)
        LOG.info("[1D %s] resolved measurements_file to %s", self.idx, resolved_path)
        return True

    def prepare_kalman_measurements(self):
        assert hasattr(self.kalman, "measurements_config")
        assert hasattr(self.kalman, "process_loaded_measurements")
        assert hasattr(self.kalman, "set_kalman_filter")

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
        _ = noisy_measurements_to_test

        sample_variance = np.nanvar(noisy_measurements, axis=0)
        measurement_noise_covariance = np.diag(sample_variance)

        self.kalman.results.times_measurements = np.cumsum(meas_model_iter_time)
        self.kalman.results.precipitation_flux_measurements = meas_model_iter_flux

        return self.kalman.set_kalman_filter(measurement_noise_covariance)

    def step(self, date_time, data_for_step):
        if hasattr(self.kalman, "step"):
            return float(self.kalman.step(date_time=date_time, pressure_head=data_for_step))

        LOG.info(
            "[1D %s] kalman class %s has no step(); using pressure_head passthrough.",
            self.idx,
            type(self.kalman).__name__,
        )
        return float(data_for_step)

    def run_loop(self, t_end, queue_name_in, queue_name_out):
        q_in = Queue(queue_name_in)
        q_out = Queue(queue_name_out)

        current_time = 0.0
        while current_time < t_end:
            msg_in = q_in.get()
            assert isinstance(msg_in, Data3DTo1D), f"Unexpected 3D->1D payload: {type(msg_in)}"
            assert msg_in.site_id == self.idx, f"Expected site_id {self.idx}, got {msg_in.site_id}"

            bottom_velocity = self.step(msg_in.date_time, msg_in.pressure_head)
            q_out.put(
                Data1DTo3D(
                    date_time=msg_in.date_time,
                    site_id=self.idx,
                    longitude=float(self.location.longitude),
                    latitude=float(self.location.latitude),
                    velocity=bottom_velocity,
                )
            )
            LOG.info("[1D %s] sent contribution=%s at date_time=%s", self.idx, bottom_velocity, msg_in.date_time)
            current_time = datetime64_to_model_time_days(msg_in.date_time)

        LOG.info("[1D %s] finished loop at t=%s (t_end=%s)", self.idx, current_time, t_end)
        return f"1D model {self.idx} done"


def model1d_worker_entry(idx, t_end, queue_name_in, queue_name_out, work_dir, config_path, location):
    model = Model1D(
        idx=idx,
        work_dir=work_dir,
        config_path=config_path,
        location=location,
    )
    return model.run_loop(t_end, queue_name_in, queue_name_out)
