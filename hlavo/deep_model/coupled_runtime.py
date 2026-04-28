from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import flopy
import numpy as np
from flopy.utils.binaryfile import HeadFile

import hlavo.deep_model.model_3d_cfg as cfg3d
import hlavo.misc.config as cfg
from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)
INVALID_HEAD_ABS_THRESHOLD = 1.0e20


class Model3DBackend:
    def __init__(self, composed: ComposedData, model_3d_cfg: dict, locations_1d) -> None:
        """
        model_3d_cfg is the 'model_3d:' part of the common config.
        """
        self.composed = composed
        self.cfg = cfg3d.Model3DCommonConfig.from_mapping(model_3d_cfg)
        self.locations_1d : int = locations_1d

        self._prepare_model_workspace()

        self.dis_file = self.composed.workdir / f"{self.cfg.sim_name}.dis"
        self.nam_file = self.composed.workdir / f"{self.cfg.sim_name}.nam"
        self.rcha_file = self.composed.workdir / f"{self.cfg.sim_name}.rcha"
        self.ic_file = self.composed.workdir / f"{self.cfg.sim_name}.ic"
        self.tdis_file = self.composed.workdir / f"{self.cfg.sim_name}.tdis"
        self.hds_file = self.composed.workdir / f"{self.cfg.sim_name}.hds"

        self.nlay, self.nrow, self.ncol = self._read_dis_dimensions(self.dis_file)
        self.lon_sw, self.lat_sw, self.lon_ne, self.lat_ne = self._read_grid_corners(self.nam_file)
        self.active_mask_3d = self._read_active_mask_3d()
        self.cell_owner = None

    def _prepare_model_workspace(self) -> None:
        src = self.cfg.model_folder
        dst = self.composed.workdir
        assert src.exists(), f"3D source model folder not found: {src}"
        assert src.is_dir(), f"3D source model folder must be a directory: {src}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        LOG.info("[3D] copied model workspace: %s -> %s", src, dst)

    @staticmethod
    def _read_dis_dimensions(dis_file: Path) -> tuple[int, int, int]:
        assert dis_file.exists(), f"DIS file not found: {dis_file}"
        text = dis_file.read_text(encoding="utf-8")
        nlay = int(re.search(r"\bNLAY\s+(\d+)", text).group(1))
        nrow = int(re.search(r"\bNROW\s+(\d+)", text).group(1))
        ncol = int(re.search(r"\bNCOL\s+(\d+)", text).group(1))
        return nlay, nrow, ncol

    @staticmethod
    def _read_grid_corners(nam_file: Path) -> tuple[float, float, float, float]:
        assert nam_file.exists(), f"NAM file not found: {nam_file}"
        lines = nam_file.read_text(encoding="utf-8").splitlines()
        sw_line = next(line for line in lines if "Grid SW corner lon/lat" in line)
        ne_line = next(line for line in lines if "Grid NE corner lon/lat" in line)
        sw_vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", sw_line)]
        ne_vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ne_line)]
        lat_sw, lon_sw = sw_vals[-2], sw_vals[-1]
        lat_ne, lon_ne = ne_vals[-2], ne_vals[-1]
        return lon_sw, lat_sw, lon_ne, lat_ne

    def _read_active_mask_3d(self) -> np.ndarray:
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(self.composed.workdir), verbosity_level=0)
        gwf = sim.get_model(self.cfg.sim_name)
        idomain = np.asarray(gwf.dis.idomain.array, dtype=int)
        assert idomain.shape == (self.nlay, self.nrow, self.ncol), "Unexpected idomain shape"
        return idomain > 0

    def _cell_centers_lon_lat(self) -> tuple[np.ndarray, np.ndarray]:
        lon = np.linspace(self.lon_sw, self.lon_ne, self.ncol, endpoint=False)
        lat = np.linspace(self.lat_sw, self.lat_ne, self.nrow, endpoint=False)
        if self.ncol > 1:
            lon = lon + 0.5 * (self.lon_ne - self.lon_sw) / self.ncol
        if self.nrow > 1:
            lat = lat + 0.5 * (self.lat_ne - self.lat_sw) / self.nrow
        return np.meshgrid(lon, lat)

    def build_cell_assignment(self) -> None:
        xx, yy = self._cell_centers_lon_lat()
        lon_1d = np.asarray([loc.longitude for loc in self.locations_1d], dtype=float)
        lat_1d = np.asarray([loc.latitude for loc in self.locations_1d], dtype=float)
        dist2 = (xx[..., None] - lon_1d[None, None, :]) ** 2 + (yy[..., None] - lat_1d[None, None, :]) ** 2
        self.cell_owner = np.argmin(dist2, axis=2)

    def _write_tdis_for_step(self, dt_days: float) -> None:
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

    def _write_rcha_array(self, recharge_array: np.ndarray) -> None:
        assert recharge_array.shape == (self.nrow, self.ncol)
        with self.rcha_file.open("w", encoding="utf-8") as handle:
            handle.write("BEGIN options\n  READASARRAYS\nEND options\n\n")
            handle.write("BEGIN period  1\n  recharge\n    INTERNAL  FACTOR  1.0\n")
            for row in recharge_array:
                handle.write(" ".join(f"{value:15.8E}" for value in row) + "\n")
            handle.write("END period  1\n")

    def _run_mf6(self) -> None:
        executable = self.cfg.exe_name
        resolved_executable = shutil.which(executable)
        assert resolved_executable is not None, f"MODFLOW executable not found: {executable}"
        LOG.info("[3D] running MODFLOW: %s (cwd=%s)", resolved_executable, self.composed.workdir)
        result = subprocess.run(
            [resolved_executable],
            cwd=self.composed.workdir,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MODFLOW 6 execution failed with exit code {result.returncode}")

    def _read_last_heads(self) -> np.ndarray:
        if self.hds_file.exists():
            hds = HeadFile(str(self.hds_file))
            times = hds.get_times()
            assert times, "No times in head file"
            return np.asarray(hds.get_data(totim=times[-1]), dtype=float)

        LOG.info("[3D] head file missing at startup, using initial conditions from %s", self.ic_file)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(self.composed.workdir), verbosity_level=0)
        gwf = sim.get_model(self.cfg.sim_name)
        heads = np.asarray(gwf.ic.strt.array, dtype=float)
        assert heads.shape == (self.nlay, self.nrow, self.ncol), "Unexpected initial head shape"
        return heads

    def sanitize_heads(self, heads: np.ndarray) -> np.ndarray:
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

    def _write_ic(self, heads: np.ndarray) -> None:
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

    def spread_recharges(self, contributions) -> np.ndarray:
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

    def heads_to_1d(self, heads: np.ndarray) -> Dict[int, float]:
        assert heads.shape == (self.nlay, self.nrow, self.ncol)
        mask = lambda idx : (self.cell_owner[None, :, :] == idx) & self.active_mask_3d
        mean_head = lambda idx: np.nanmean(heads[mask(idx)])
        return {site_id: mean_head(idx) for idx, site_id in self.locations_1d}

    def initial_heads_to_1d(self) -> Dict[int, float]:
        return self.heads_to_1d(self.sanitize_heads(self._read_last_heads()))

    def choose_dt(self, current_time: np.datetime64['s'], t_end: np.datetime64['s']) -> np.timedelta64['s']:
        remaining = t_end - current_time
        return max(min(self.cfg.time_step_days, remaining), np.timedelta64(1, 's'))

    def model_step(self, dt: float, contributions) -> np.ndarray:
        recharge = self.spread_recharges(contributions)
        self._write_rcha_array(recharge)
        self._write_tdis_for_step(dt)
        self._run_mf6()
        heads = self.sanitize_heads(self._read_last_heads())
        self._write_ic(heads)
        return self.heads_to_1d(heads)
