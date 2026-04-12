#!/usr/bin/env python3

"""
Author: Jan Březina

1. It must be tested and split into 1D and 3D part.
The setup should be implemented within deep_model directory.

2. Proper data must be passed through the queues:
From deep model to 1D models:
- end time of the next time interval
- timestep to use
- sucction pressure at the bottom of the 1D model (and top of the 3D model)
From 1D models to the deep model:
- time series, velocity at the bottom

The times should be passed as the np.datetime64[ms] objects in order to
keep relation to the date time series of the measurements.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import *

import attrs
import flopy
import numpy as np
import yaml
from dask.distributed import Client, LocalCluster, Queue, get_client
from flopy.utils.binaryfile import HeadFile

from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.ingress.moist_profile.load_data import load_data
from hlavo.kalman.kalman import KalmanFilter
from hlavo.misc.class_resolve import resolve_named_class

LOG = logging.getLogger(__name__)
INVALID_HEAD_ABS_THRESHOLD = 1.0e20
TIME_ORIGIN = np.datetime64("2000-01-01T00:00:00", "ms")
MILLISECONDS_PER_DAY = 86_400_000.0

# AGENT: folloing manual calculations should be done by standard numpy datetime64 arithmetic,
# either explain in comment why it is necessary or replace it (and associated constants)
def model_time_days_to_datetime64(model_time_days: float) -> np.datetime64:
    milliseconds = int(round(float(model_time_days) * MILLISECONDS_PER_DAY))
    return TIME_ORIGIN + np.timedelta64(milliseconds, "ms")


def datetime64_to_model_time_days(date_time: np.datetime64) -> float:
    delta_ms = (np.datetime64(date_time, "ms") - TIME_ORIGIN) / np.timedelta64(1, "ms")
    return float(delta_ms) / MILLISECONDS_PER_DAY


class Model1DMock:
    # AGENT: this is a placeholder for the actual 1D model implementation, which should be implemented in the future.
    # but we will keep both implementation Model1DMock and Model1DKalman for testing and comparison purposes.
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
            self.kalman = KalmanFilter.from_config(self.work_dir, main_cfg_data, verbose=False)
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
        for (time_prec, precipitation) in meas_model_iter_flux:
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
            # TEMPORARY_MOCK_RECHARGE: replace this random value with real 1D-model
            # recharge result once 1D computation output is wired correctly.
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


Model1D = Model1DMock


def model1d_worker_entry(idx, t_end, queue_name_in, queue_name_out, work_dir, kalman_config, location):
    with Path(kalman_config).open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    model_1d_cfg = config_data.get("model_1d", {})
    assert isinstance(model_1d_cfg, dict), "model_1d config must be a mapping"
    model_class_name = str(model_1d_cfg.get("class_name", "Model1DMock"))
    model_class = resolve_named_class(model_class_name, ("hlavo.composed.composed_model_mock",))

    model = model_class(
        idx=idx,
        initial_state=0.0,
        work_dir=work_dir,
        kalman_config=kalman_config,
        location=location,
    )
    return model.run_loop(t_end, queue_name_in, queue_name_out)


@attrs.define(frozen=True)
class Model1DLocation:
    idx: int
    longitude: float
    latitude: float


@attrs.define(frozen=True)
class Model3DConfig:
    model_name: str
    model_folder: Path
    work_folder: Path
    time_step_days: float
    executable: str = "mf6"
    model_class_name: str = "Model3D"
    total_time_days: float | None = None
    n_steps: int | None = None

    @classmethod
    def from_yaml(cls, config_data: dict, work_dir: Path) -> "Model3DConfig":
        model_3d = config_data.get("model_3d", {})
        assert isinstance(model_3d, dict), "model_3d config must be a mapping"
        model_name = str(model_3d.get("name", "uhelna"))

        model_folder = (work_dir / "model_with_mine").resolve()
        assert model_folder.exists(), f"model_3d.folder does not exist: {model_folder}"
        assert model_folder.is_dir(), f"model_3d.folder must be a directory: {model_folder}"

        work_folder = (work_dir / "model_with_mine_work").resolve()
        assert work_folder != model_folder, "model_3d.work_folder must be different from model_3d.folder"

        time_step_days = float(model_3d.get("time_step_days", 5.0))
        assert time_step_days > 0.0, "model_3d.time_step_days must be > 0"

        executable = str(model_3d.get("executable", "mf6"))
        model_class_name = str(model_3d.get("class_name", "Model3D"))
        total_time_days = None if "total_time_days" not in model_3d else float(model_3d["total_time_days"])
        n_steps = None if "n_steps" not in model_3d else int(model_3d["n_steps"])

        return cls(
            model_name=model_name,
            model_folder=model_folder,
            work_folder=work_folder,
            time_step_days=time_step_days,
            executable=executable,
            model_class_name=model_class_name,
            total_time_days=total_time_days,
            n_steps=n_steps,
        )

class Model3D:
    def __init__(self, n_1d, model_3d_cfg: Model3DConfig, locations_1d, initial_time=0.0):
        self.n_1d = n_1d
        self.cfg = model_3d_cfg
        self.locations_1d = locations_1d
        self.time: float = 0.0
        # simulation time in days from start of simulated interval

        self._prepare_model_workspace()

        self.dis_file = self.cfg.work_folder / f"{self.cfg.model_name}.dis"
        self.nam_file = self.cfg.work_folder / f"{self.cfg.model_name}.nam"
        self.rcha_file = self.cfg.work_folder / f"{self.cfg.model_name}.rcha"
        self.ic_file = self.cfg.work_folder / f"{self.cfg.model_name}.ic"
        self.tdis_file = self.cfg.work_folder / f"{self.cfg.model_name}.tdis"
        self.hds_file = self.cfg.work_folder / f"{self.cfg.model_name}.hds"

        self.nlay, self.nrow, self.ncol = self._read_dis_dimensions(self.dis_file)
        self.lon_sw, self.lat_sw, self.lon_ne, self.lat_ne = self._read_grid_corners(self.nam_file)
        self.active_mask_3d = self._read_active_mask_3d()
        self.cell_owner = None

    def resolve_t_end(self) -> float:
        if self.cfg.total_time_days is not None:
            t_end = self.cfg.total_time_days
        elif self.cfg.n_steps is not None:
            t_end = self.cfg.time_step_days * float(self.cfg.n_steps)
        else:
            t_end = self.cfg.time_step_days
        assert t_end > 0.0, "model_3d total simulation time must be > 0"
        return t_end

    def _prepare_model_workspace(self):
        src = self.cfg.model_folder
        dst = self.cfg.work_folder
        assert src.exists(), f"3D source model folder not found: {src}"
        assert src.is_dir(), f"3D source model folder must be a directory: {src}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        LOG.info("[3D] copied model workspace: %s -> %s", src, dst)

    @staticmethod
    def _read_dis_dimensions(dis_file: Path):
        assert dis_file.exists(), f"DIS file not found: {dis_file}"
        text = dis_file.read_text(encoding="utf-8")
        nlay = int(re.search(r"\bNLAY\s+(\d+)", text).group(1))
        nrow = int(re.search(r"\bNROW\s+(\d+)", text).group(1))
        ncol = int(re.search(r"\bNCOL\s+(\d+)", text).group(1))
        return nlay, nrow, ncol

    @staticmethod
    def _read_grid_corners(nam_file: Path):
        assert nam_file.exists(), f"NAM file not found: {nam_file}"
        lines = nam_file.read_text(encoding="utf-8").splitlines()
        sw_line = next(line for line in lines if "Grid SW corner lon/lat" in line)
        ne_line = next(line for line in lines if "Grid NE corner lon/lat" in line)

        sw_vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", sw_line)]
        ne_vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ne_line)]

        # in comments values are written as lat, lon
        lat_sw, lon_sw = sw_vals[-2], sw_vals[-1]
        lat_ne, lon_ne = ne_vals[-2], ne_vals[-1]
        return lon_sw, lat_sw, lon_ne, lat_ne

    def _cell_centers_lon_lat(self):
        lon = np.linspace(self.lon_sw, self.lon_ne, self.ncol, endpoint=False)
        lat = np.linspace(self.lat_sw, self.lat_ne, self.nrow, endpoint=False)
        if self.ncol > 1:
            lon = lon + 0.5 * (self.lon_ne - self.lon_sw) / self.ncol
        if self.nrow > 1:
            lat = lat + 0.5 * (self.lat_ne - self.lat_sw) / self.nrow
        xx, yy = np.meshgrid(lon, lat)
        return xx, yy

    def _build_cell_assignment(self):
        xx, yy = self._cell_centers_lon_lat()
        lon_1d = np.asarray([loc.longitude for loc in self.locations_1d], dtype=float)
        lat_1d = np.asarray([loc.latitude for loc in self.locations_1d], dtype=float)
        dist2 = (xx[..., None] - lon_1d[None, None, :]) ** 2 + (yy[..., None] - lat_1d[None, None, :]) ** 2
        self.cell_owner = np.argmin(dist2, axis=2)

    def _write_tdis_for_step(self, dt_days):
        text = (
            "BEGIN options\n"
            "  TIME_UNITS  days\n"
            "END options\n\n"
            "BEGIN dimensions\n"
            "  NPER  1\n"
            "END dimensions\n\n"
            "BEGIN perioddata\n"
            f"  {dt_days:.8f}  1  1.00000000\n"
            "END perioddata\n"
        )
        self.tdis_file.write_text(text, encoding="utf-8")

    def _read_active_mask_3d(self):
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(self.cfg.work_folder), verbosity_level=0)
        gwf = sim.get_model(self.cfg.model_name)
        idomain = np.asarray(gwf.dis.idomain.array, dtype=int)
        assert idomain.shape == (self.nlay, self.nrow, self.ncol), "Unexpected idomain shape"
        return idomain > 0

    def _write_rcha_array(self, recharge_array):
        assert recharge_array.shape == (self.nrow, self.ncol)
        with self.rcha_file.open("w", encoding="utf-8") as handle:
            handle.write("BEGIN options\n  READASARRAYS\nEND options\n\n")
            handle.write("BEGIN period  1\n  recharge\n    INTERNAL  FACTOR  1.0\n")
            for row in recharge_array:
                handle.write(" ".join(f"{value:15.8E}" for value in row) + "\n")
            handle.write("END period  1\n")

    def _run_mf6(self):
        executable = self.cfg.executable
        resolved_executable = shutil.which(executable)
        assert resolved_executable is not None, f"MODFLOW executable not found: {executable}"

        LOG.info("[3D] running MODFLOW: %s (cwd=%s)", resolved_executable, self.cfg.work_folder)
        result = subprocess.run(
            [resolved_executable],
            cwd=self.cfg.work_folder,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MODFLOW 6 execution failed with exit code {result.returncode}")

    def _read_last_heads(self):
        if self.hds_file.exists():
            hds = HeadFile(str(self.hds_file))
            times = hds.get_times()
            assert times, "No times in head file"
            return np.asarray(hds.get_data(totim=times[-1]), dtype=float)

        LOG.info("[3D] head file missing at startup, using initial conditions from %s", self.ic_file)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(self.cfg.work_folder), verbosity_level=0)
        gwf = sim.get_model(self.cfg.model_name)
        heads = np.asarray(gwf.ic.strt.array, dtype=float)
        assert heads.shape == (self.nlay, self.nrow, self.ncol), "Unexpected initial head shape"
        return heads

    def _sanitize_heads(self, heads):
        assert heads.shape == (self.nlay, self.nrow, self.ncol)
        valid_mask = np.isfinite(heads) & (np.abs(heads) < INVALID_HEAD_ABS_THRESHOLD)
        valid_count = int(np.sum(valid_mask))
        assert valid_count > 0, "No valid head values found in MODFLOW output"

        fill_value = float(np.median(heads[valid_mask]))
        cleaned = np.where(valid_mask, heads, fill_value)
        invalid_count = int(heads.size - valid_count)
        if invalid_count > 0:
            LOG.warning(
                "[3D] sanitized %s invalid head cells (|h| >= %s or non-finite), fill_value=%s",
                invalid_count,
                INVALID_HEAD_ABS_THRESHOLD,
                fill_value,
            )
        return cleaned

    def _write_ic(self, heads):
        assert heads.shape == (self.nlay, self.nrow, self.ncol)
        with self.ic_file.open("w", encoding="utf-8") as handle:
            handle.write("BEGIN options\nEND options\n\n")
            handle.write("BEGIN griddata\n")
            handle.write("  strt  LAYERED\n")
            for layer in range(self.nlay):
                handle.write("    INTERNAL  FACTOR  1.0\n")
                for row in heads[layer]:
                    handle.write(" ".join(f"{value:15.8f}" for value in row) + "\n")
            handle.write("END griddata\n")

    def _spread_recharges(self, contributions):
        recharge = np.zeros((self.nrow, self.ncol), dtype=float)
        raw = np.asarray(contributions, dtype=float)
        LOG.info(
            "[3D] recharge input raw_min=%s raw_max=%s",
            float(np.nanmin(raw)),
            float(np.nanmax(raw)),
        )
        for idx, value in enumerate(raw):
            recharge[self.cell_owner == idx] = float(value)
        return recharge

    def _heads_to_1d(self, heads):
        assert heads.shape == (self.nlay, self.nrow, self.ncol)
        values = np.zeros(self.n_1d, dtype=float)
        for idx in range(self.n_1d):
            mask = (self.cell_owner[None, :, :] == idx) & self.active_mask_3d
            if np.any(mask):
                values[idx] = float(np.nanmean(heads[mask]))
        return values

    def choose_dt(self, t_end):
        remaining = t_end - self.time
        return max(min(self.cfg.time_step_days, remaining), 0.0)

    def model_step(self, dt, contributions):
        recharge = self._spread_recharges(contributions)
        self._write_rcha_array(recharge)
        self._write_tdis_for_step(dt)
        self._run_mf6()

        heads = self._sanitize_heads(self._read_last_heads())
        self._write_ic(heads)

        heads_to_1d = self._heads_to_1d(heads)
        return heads_to_1d

    def run_loop(
        self,
        t_interval: float | Tuple[np.datetime64, np.datetime64],
        queue_names_out_to_1d: List[str],
        queue_name_in_from_1d: str,
    ):
        if isinstance(t_interval, tuple):
            _, t_end_raw = t_interval
            t_end = datetime64_to_model_time_days(t_end_raw)
        else:
            t_end = float(t_interval)
        q_3d_to_1d = [Queue(name) for name in queue_names_out_to_1d]
        q_1d_to_3d = Queue(queue_name_in_from_1d)

        self._build_cell_assignment()

        heads = self._sanitize_heads(self._read_last_heads())
        initial_heads = self._heads_to_1d(heads)
        for i in range(self.n_1d):
            #
            msg_out = Data3DTo1D(
                date_time=model_time_days_to_datetime64(self.time),
                site_id=i,
                pressure_head=float(initial_heads[i]),
            )
            q_3d_to_1d[i].put(msg_out)
            LOG.info("[3D] startup push -> 1D %s: date_time=%s, head=%s", i, msg_out.date_time, initial_heads[i])

        while self.time < t_end:
            dt = self.choose_dt(t_end)
            if dt <= 0.0:
                LOG.info("[3D] dt <= 0, stopping to avoid infinite loop.")
                break

            target_time = self.time + dt
            LOG.info("[3D] === Step: t=%s -> t=%s ===", self.time, target_time)

            contributions = [None] * self.n_1d
            received = 0
            while received < self.n_1d:
                msg_in = q_1d_to_3d.get()
                assert isinstance(msg_in, Data1DTo3D), f"Unexpected 1D->3D payload: {type(msg_in)}"
                idx = int(msg_in.site_id)
                assert 0 <= idx < self.n_1d, f"site_id out of range: {idx}"
                LOG.info("[3D] received from 1D %s: date_time=%s, recharge=%s", idx, msg_in.date_time, msg_in.velocity)
                contributions[idx] = float(msg_in.velocity)
                received += 1

            heads_to_1d = self.model_step(dt, contributions)
            for i in range(self.n_1d):
                msg_out = Data3DTo1D(
                    date_time=model_time_days_to_datetime64(target_time),
                    site_id=i,
                    pressure_head=float(heads_to_1d[i]),
                )
                q_3d_to_1d[i].put(msg_out)
                LOG.info("[3D] send head -> 1D %s: date_time=%s, head=%s", i, msg_out.date_time, heads_to_1d[i])

            self.time = target_time

        LOG.info("[3D] finished time loop at t=%s (t_end=%s)", self.time, t_end)
        return self.time


def setup_models(work_dir, config_path, client):
    work_dir = Path(work_dir).resolve()
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    model_3d_cfg = Model3DConfig.from_yaml(config_data, work_dir)
    locations_1d = _parse_locations(config_data)

    n_1d = len(locations_1d)

    queue_names_3d_to_1d = []
    futures_1d = []

    queue_name_1d_to_3d = "q-1d-to-3d"
    Queue(queue_name_1d_to_3d, client=client)

    model_3d_class = resolve_named_class(model_3d_cfg.model_class_name, ("hlavo.composed.composed_model_mock",))
    model_3d = model_3d_class(n_1d=n_1d, model_3d_cfg=model_3d_cfg, locations_1d=locations_1d)
    t_end = model_3d.resolve_t_end()

    for i in range(n_1d):
        q_name_3d_to_1d = f"q-3d-to-1d-{i}"
        Queue(q_name_3d_to_1d, client=client)
        queue_names_3d_to_1d.append(q_name_3d_to_1d)

        fut = client.submit(
            model1d_worker_entry,
            i,
            t_end,
            q_name_3d_to_1d,
            queue_name_1d_to_3d,
            work_dir,
            config_path,
            locations_1d[i],
            pure=False,
        )
        futures_1d.append(fut)
        LOG.info("[SETUP] Submitted Model1D idx=%s", i)
    final_state_3d = model_3d.run_loop(
        t_end,
        queue_names_out_to_1d=queue_names_3d_to_1d,
        queue_name_in_from_1d=queue_name_1d_to_3d,
    )

    LOG.info("[SETUP] Waiting for all 1D models to finish...")
    results_1d = [f.result() for f in futures_1d]
    LOG.info("[SETUP] 1D model results: %s", results_1d)

    return final_state_3d


def _parse_locations(config_data):
    model_1d_cfg = config_data.get("model_1d", {})
    assert isinstance(model_1d_cfg, dict), "model_1d must be a mapping"
    raw_locations = model_1d_cfg.get("sites", [])
    assert isinstance(raw_locations, list), "model_1d.sites must be a list"
    assert raw_locations, "model_1d.sites must not be empty"

    locations = []
    for idx, item in enumerate(raw_locations):
        assert isinstance(item, dict), "Each model_1d item must be a mapping"
        locations.append(
            Model1DLocation(
                idx=idx,
                longitude=float(item["longitude"]),
                latitude=float(item["latitude"]),
            )
        )
    return locations


def run_simulation(work_dir: Path, config_path: Path) -> float:
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    try:
        final_state = setup_models(work_dir, config_path, client)
        LOG.info("[MAIN] Final 3D time: %s", final_state)
        return float(final_state)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    raise SystemExit("Use hlavo/main.py simulate <config_file> [-w <workdir>]")
