from __future__ import annotations

import logging
from pathlib import Path

import attrs
import numpy as np
import zarr_fuse

import hlavo.deep_model.model_3d_cfg as cfg3d
import hlavo.misc.config as cfg
from hlavo.deep_model.qgis_reader import ModelGeometry

LOG = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[2]
DEFAULT_WELLS_SCHEMA_PATH = ROOT_PATH / "hlavo/ingress/well_data/wells_schema.yaml"


@attrs.define(frozen=True)
class PumpingWell:
    well_id: str
    longitude: float
    latitude: float
    layer: int
    row: int
    col: int
    rates_m3_per_day: tuple[float, ...]

    def stress_period_data(self) -> dict[int, list[tuple[int, int, int, float]]]:
        return {
            iper: [(self.layer, self.row, self.col, float(rate))]
            for iper, rate in enumerate(self.rates_m3_per_day)
        }


def _store_kwargs(store_url: str | Path | None) -> dict[str, str]:
    if store_url is None:
        return {}
    return {"STORE_URL": str(store_url)}


def _load_pumping_wells_cfg(config_source: Path | dict) -> dict | None:
    raw, _ = cfg.load_config(config_source)
    model_3d_raw = cfg3d.resolve_model_3d_section(raw)
    pumping_raw = model_3d_raw.get("pumping_wells")
    if pumping_raw is None:
        return None
    assert isinstance(pumping_raw, dict), "model_3d.pumping_wells must be a mapping"
    return pumping_raw


def _period_rates_from_draws(
    *,
    draw_dates: np.ndarray,
    draw_values: np.ndarray,
    stress_periods_days: tuple[float, ...],
) -> tuple[float, ...]:
    order = np.argsort(draw_dates.astype("datetime64[s]").astype(np.int64))
    sorted_values = np.asarray(draw_values, dtype=float)[order]
    assert sorted_values.ndim == 1, "cum_draw series must be 1D"
    assert sorted_values.size == len(stress_periods_days), (
        "Basic pumping-well scheduling requires one cum_draw value per stress period"
    )
    rates = []
    for period_days, cum_draw in zip(stress_periods_days, sorted_values.tolist()):
        assert period_days > 0.0, "stress_periods_days values must be > 0"
        rates.append(-float(cum_draw) / float(period_days))
    return tuple(rates)


def _nearest_cell(
    *,
    geometry: ModelGeometry,
    longitude: float,
    latitude: float,
) -> tuple[int, int, int]:
    xy_local = geometry.lonlat_to_xy_local(np.asarray([[longitude, latitude]], dtype=float))[0]
    x_value = float(xy_local[0])
    y_value = float(xy_local[1])
    x_index = int(np.floor((x_value - float(geometry.grid.origin[0])) / float(geometry.grid.step[0])))
    y_index = int(np.floor((y_value - float(geometry.grid.origin[1])) / float(geometry.grid.step[1])))
    nx = int(geometry.grid.el_dims[0])
    ny = int(geometry.grid.el_dims[1])
    assert 0 <= x_index < nx, f"Well longitude/latitude maps outside model X extent: {(longitude, latitude)}"
    assert 0 <= y_index < ny, f"Well longitude/latitude maps outside model Y extent: {(longitude, latitude)}"
    row = ny - 1 - y_index
    col = x_index
    assert bool(geometry.active_mask[row, col]), f"Well {(longitude, latitude)} maps to inactive cell {(row, col)}"
    return (0, row, col)


def load_pumping_wells(
    *,
    config_source: Path | dict,
    common: cfg3d.Model3DCommonConfig,
    geometry: ModelGeometry,
) -> tuple[PumpingWell, ...]:
    pumping_cfg = _load_pumping_wells_cfg(config_source)
    if pumping_cfg is None:
        return ()

    schema_path = Path(str(pumping_cfg.get("schema_path", DEFAULT_WELLS_SCHEMA_PATH)))
    root = zarr_fuse.open_store(schema_path, **_store_kwargs(pumping_cfg.get("store_url")))
    dataset = root["Uhelna"]["water_draw"].dataset
    requested_well_ids = pumping_cfg.get("well_ids")
    if requested_well_ids is None:
        well_ids = [str(value) for value in dataset["well_id"].values.tolist()]
    else:
        assert isinstance(requested_well_ids, (list, tuple)), "model_3d.pumping_wells.well_ids must be a list"
        well_ids = [str(value) for value in requested_well_ids]

    pumping_wells: list[PumpingWell] = []
    for well_id in well_ids:
        item = dataset.sel(well_id=well_id)
        cum_draw = np.asarray(item["cum_draw"], dtype=float).reshape(-1)
        assert cum_draw.size > 0, f"No water_draw values found for pumping well {well_id}"
        longitude = float(np.asarray(item["longitude"], dtype=float).reshape(-1)[0])
        latitude = float(np.asarray(item["latitude"], dtype=float).reshape(-1)[0])
        draw_dates = np.asarray(item["date"].values).reshape(-1)
        layer, row, col = _nearest_cell(
            geometry=geometry,
            longitude=longitude,
            latitude=latitude,
        )
        rates = _period_rates_from_draws(
            draw_dates=draw_dates,
            draw_values=cum_draw,
            stress_periods_days=common.stress_periods_days,
        )
        pumping_wells.append(
            PumpingWell(
                well_id=well_id,
                longitude=longitude,
                latitude=latitude,
                layer=layer,
                row=row,
                col=col,
                rates_m3_per_day=rates,
            )
        )

    LOG.info("Loaded %s pumping wells from %s", len(pumping_wells), schema_path)
    return tuple(pumping_wells)
