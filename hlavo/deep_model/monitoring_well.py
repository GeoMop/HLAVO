from __future__ import annotations

import logging
from pathlib import Path

import attrs
import numpy as np
import polars as pl
import zarr_fuse

import hlavo.deep_model.model_3d_cfg as cfg3d
import hlavo.misc.config as cfg
from hlavo.composed.writer import DEFAULT_SIMULATION_SCHEMA_PATH, DEFAULT_WELLS_SCHEMA_PATH
from hlavo.deep_model.qgis_reader import ModelGeometry

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class MonitoringWell:
    well_id: str
    longitude: float
    latitude: float
    z: float | None
    interval_min: float | None
    interval_max: float | None
    layer_ids: tuple[int, ...]
    row: int
    col: int

    def water_level_from_head(self, head: np.ndarray) -> float:
        values = np.asarray([float(head[layer, self.row, self.col]) for layer in self.layer_ids], dtype=float)
        valid = values[np.isfinite(values)]
        assert valid.size > 0, f"No finite head values available for monitoring well {self.well_id}"
        return float(np.mean(valid))


def _store_kwargs(store_url: str | Path | None) -> dict[str, str]:
    if store_url is None:
        return {}
    return {"STORE_URL": str(store_url)}


def _load_monitoring_wells_cfg(config_source: Path | dict) -> dict | None:
    raw, _ = cfg.load_config(config_source)
    model_3d_raw = cfg3d.resolve_model_3d_section(raw)
    monitoring_raw = model_3d_raw.get("monitoring_wells")
    if monitoring_raw is None:
        return None
    assert isinstance(monitoring_raw, dict), "model_3d.monitoring_wells must be a mapping"
    return monitoring_raw


def _nearest_row_col(
    *,
    geometry: ModelGeometry,
    longitude: float,
    latitude: float,
) -> tuple[int, int]:
    xy_local = geometry.lonlat_to_xy_local(np.asarray([[longitude, latitude]], dtype=float))[0]
    x_value = float(xy_local[0])
    y_value = float(xy_local[1])
    x_index = int(np.floor((x_value - float(geometry.grid.origin[0])) / float(geometry.grid.step[0])))
    y_index = int(np.floor((y_value - float(geometry.grid.origin[1])) / float(geometry.grid.step[1])))
    nx = int(geometry.grid.el_dims[0])
    ny = int(geometry.grid.el_dims[1])
    assert 0 <= x_index < nx, f"Monitoring well maps outside model X extent: {(longitude, latitude)}"
    assert 0 <= y_index < ny, f"Monitoring well maps outside model Y extent: {(longitude, latitude)}"
    row = ny - 1 - y_index
    col = x_index
    assert bool(geometry.active_mask[row, col]), (
        f"Monitoring well {(longitude, latitude)} maps to inactive cell {(row, col)}"
    )
    return row, col


def _top_active_layer(*, geometry: ModelGeometry, row: int, col: int) -> int:
    idomain = np.broadcast_to(np.asarray(geometry.active_mask, dtype=bool), np.asarray(geometry.botm).shape)
    active_layers = np.flatnonzero(idomain[:, row, col])
    assert active_layers.size > 0, f"No active model layer found at monitoring well cell {(row, col)}"
    return int(active_layers[0])


def _layers_from_screen(
    *,
    geometry: ModelGeometry,
    row: int,
    col: int,
    z: float | None,
    interval_min: float | None,
    interval_max: float | None,
) -> tuple[int, ...]:
    if z is None or interval_min is None or interval_max is None:
        return (_top_active_layer(geometry=geometry, row=row, col=col),)

    screen_top = float(z) - float(interval_min)
    screen_bottom = float(z) - float(interval_max)
    if screen_bottom > screen_top:
        screen_top, screen_bottom = screen_bottom, screen_top

    top = np.asarray(geometry.top, dtype=float)
    botm = np.asarray(geometry.botm, dtype=float)
    idomain = np.broadcast_to(np.asarray(geometry.active_mask, dtype=bool), botm.shape)

    layer_ids: list[int] = []
    for layer in range(botm.shape[0]):
        if not bool(idomain[layer, row, col]):
            continue
        layer_top = float(top[row, col]) if layer == 0 else float(botm[layer - 1, row, col])
        layer_bottom = float(botm[layer, row, col])
        overlaps = min(layer_top, screen_top) > max(layer_bottom, screen_bottom)
        if overlaps:
            layer_ids.append(layer)

    if layer_ids:
        return tuple(layer_ids)

    screen_mid = 0.5 * (screen_top + screen_bottom)
    active_layers = np.flatnonzero(idomain[:, row, col])
    assert active_layers.size > 0, f"No active model layer found at monitoring well cell {(row, col)}"
    distances = []
    for layer in active_layers.tolist():
        layer_top = float(top[row, col]) if layer == 0 else float(botm[layer - 1, row, col])
        layer_bottom = float(botm[layer, row, col])
        distances.append(abs(0.5 * (layer_top + layer_bottom) - screen_mid))
    return (int(active_layers[int(np.argmin(np.asarray(distances, dtype=float)))]),)


def load_monitoring_wells(
    *,
    config_source: Path | dict,
    geometry: ModelGeometry,
) -> tuple[MonitoringWell, ...]:
    monitoring_cfg = _load_monitoring_wells_cfg(config_source)
    if monitoring_cfg is None:
        return ()

    schema_path = Path(str(monitoring_cfg.get("schema_path", DEFAULT_WELLS_SCHEMA_PATH)))
    root = zarr_fuse.open_store(schema_path, **_store_kwargs(monitoring_cfg.get("store_url")))
    dataset = root["Uhelna"]["water_levels"].dataset
    requested_well_ids = monitoring_cfg.get("well_ids")
    if requested_well_ids is None:
        well_ids = [str(value) for value in dataset["well_id"].values.tolist()]
    else:
        assert isinstance(requested_well_ids, (list, tuple)), "model_3d.monitoring_wells.well_ids must be a list"
        well_ids = [str(value) for value in requested_well_ids]

    monitoring_wells: list[MonitoringWell] = []
    for well_id in well_ids:
        item = dataset.sel(well_id=well_id)
        longitude = float(np.asarray(item["longitude"], dtype=float).reshape(-1)[0])
        latitude = float(np.asarray(item["latitude"], dtype=float).reshape(-1)[0])
        z_values = np.asarray(item["Z"], dtype=float).reshape(-1) if "Z" in item else np.asarray([np.nan])
        interval_min_values = (
            np.asarray(item["interval_min"], dtype=float).reshape(-1) if "interval_min" in item else np.asarray([np.nan])
        )
        interval_max_values = (
            np.asarray(item["interval_max"], dtype=float).reshape(-1) if "interval_max" in item else np.asarray([np.nan])
        )
        z = None if np.all(np.isnan(z_values)) else float(z_values[np.flatnonzero(~np.isnan(z_values))[0]])
        interval_min = (
            None if np.all(np.isnan(interval_min_values)) else float(interval_min_values[np.flatnonzero(~np.isnan(interval_min_values))[0]])
        )
        interval_max = (
            None if np.all(np.isnan(interval_max_values)) else float(interval_max_values[np.flatnonzero(~np.isnan(interval_max_values))[0]])
        )
        row, col = _nearest_row_col(geometry=geometry, longitude=longitude, latitude=latitude)
        monitoring_wells.append(
            MonitoringWell(
                well_id=well_id,
                longitude=longitude,
                latitude=latitude,
                z=z,
                interval_min=interval_min,
                interval_max=interval_max,
                layer_ids=_layers_from_screen(
                    geometry=geometry,
                    row=row,
                    col=col,
                    z=z,
                    interval_min=interval_min,
                    interval_max=interval_max,
                ),
                row=row,
                col=col,
            )
        )

    LOG.info("Loaded %s monitoring wells from %s", len(monitoring_wells), schema_path)
    return tuple(monitoring_wells)


def monitoring_output_times(
    *,
    config_source: Path | dict,
    stress_periods_days: tuple[float, ...],
) -> tuple[np.datetime64, ...]:
    raw, _ = cfg.load_config(config_source)
    start_raw = raw.get("start_datetime")
    if start_raw is None:
        model_3d = cfg3d.resolve_model_3d_section(raw)
        common_raw = model_3d.get("common", {})
        if isinstance(common_raw, dict):
            start_raw = common_raw.get("start_datetime")
    assert start_raw is not None, (
        "Monitoring-well export requires start_datetime in the root config or model_3d.common"
    )
    current = np.datetime64(str(start_raw), "m")
    result: list[np.datetime64] = []
    for period_days in stress_periods_days:
        minutes = int(round(float(period_days) * 24.0 * 60.0))
        current = current + np.timedelta64(minutes, "m")
        result.append(current)
    return tuple(result)


def write_monitoring_well_predictions(
    *,
    config_source: Path | dict,
    monitoring_wells: tuple[MonitoringWell, ...],
    date_times: tuple[np.datetime64, ...],
    heads_by_period: tuple[np.ndarray, ...],
) -> None:
    monitoring_cfg = _load_monitoring_wells_cfg(config_source)
    if monitoring_cfg is None or not monitoring_wells:
        return

    assert len(date_times) == len(heads_by_period), "date_times and heads_by_period length mismatch"
    simulation_schema_path = Path(
        str(monitoring_cfg.get("simulation_schema_path", DEFAULT_SIMULATION_SCHEMA_PATH))
    )
    root = zarr_fuse.open_store(
        simulation_schema_path,
        **_store_kwargs(monitoring_cfg.get("simulation_store_url")),
    )

    calibration = np.datetime_as_string(np.datetime64(date_times[0], "m"), unit="m", timezone="UTC")
    rows: list[dict] = []
    for date_time, head in zip(date_times, heads_by_period):
        date_label = np.datetime_as_string(np.datetime64(date_time, "m"), unit="m", timezone="UTC")
        for well in monitoring_wells:
            water_level = well.water_level_from_head(head)
            rows.append(
                {
                    "date_time": date_label,
                    "well_id": well.well_id,
                    "calibration": calibration,
                    "water_level": water_level,
                    "water_depth": np.nan if well.z is None else float(well.z - water_level),
                    "longitude": well.longitude,
                    "latitude": well.latitude,
                    "Z": np.nan if well.z is None else float(well.z),
                }
            )
    root["Uhelna"]["well_prediction"].update(pl.DataFrame(rows))
    LOG.info("Wrote %s monitoring-well predictions to %s", len(rows), simulation_schema_path)
