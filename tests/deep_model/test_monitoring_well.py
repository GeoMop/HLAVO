from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import zarr_fuse
from pyproj import Transformer

from hlavo.deep_model.monitoring_well import (
    load_monitoring_wells,
    monitoring_output_times,
    write_monitoring_well_predictions,
)
from hlavo.deep_model.qgis_reader import Grid, ModelGeometry


def _write_wells_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  water_levels:
    VARS:
      water_level:
        type: float
        coords: ["date_time", "well_id"]
      water_depth:
        type: float
        coords: ["date_time", "well_id"]
      longitude:
        type: float
        coords: ["well_id"]
      latitude:
        type: float
        coords: ["well_id"]
      Z:
        type: float
        coords: ["well_id"]
      interval_min:
        type: float
        coords: ["well_id", "interval_num_from_top"]
      interval_max:
        type: float
        coords: ["well_id", "interval_num_from_top"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      well_id:
        type: str[16]
        sorted: []
      interval_num_from_top:
        type: str[16]
        sorted: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_simulation_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  well_prediction:
    VARS:
      water_level:
        type: float
        coords: ["date_time", "well_id", "calibration"]
      water_depth:
        type: float
        coords: ["date_time", "well_id", "calibration"]
      longitude:
        type: float
        coords: ["well_id"]
      latitude:
        type: float
        coords: ["well_id"]
      Z:
        type: float
        coords: ["well_id"]
    COORDS:
      date_time:
        unit: {{ tick: "m", tz: "UTC" }}
      well_id:
        type: str[16]
        sorted: []
      calibration:
        type: str[16]
        sorted: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_wells_store(
    schema_path: Path,
    store_path: Path,
    *,
    longitude: float,
    latitude: float,
) -> None:
    _write_wells_schema(schema_path, store_path)
    root = zarr_fuse.open_store(schema_path, STORE_URL=str(store_path))
    root["Uhelna"]["water_levels"].update(
        pl.DataFrame(
            [
                {
                    "date_time": "2025-03-01T00:00",
                    "well_id": "MW1",
                    "interval_num_from_top": "0",
                    "water_level": 96.0,
                    "water_depth": 24.0,
                    "longitude": longitude,
                    "latitude": latitude,
                    "Z": 120.0,
                    "interval_min": 10.0,
                    "interval_max": 30.0,
                }
            ]
        )
    )


def _geometry() -> ModelGeometry:
    to_jtsk = Transformer.from_crs(4326, 5514, always_xy=True)
    origin_x, origin_y = to_jtsk.transform(14.0, 50.0)
    boundary_origin = np.asarray([origin_x, origin_y], dtype=float)
    return ModelGeometry(
        boundary_origin=boundary_origin,
        grid=Grid(
            origin=np.asarray([0.0, 0.0, 90.0], dtype=float),
            step=np.asarray([10.0, 10.0, 10.0], dtype=float),
            el_dims=np.asarray([2, 1, 3], dtype=int),
        ),
        rasters=(),
        active_mask=np.asarray([[True, True]], dtype=bool),
        top=np.asarray([[120.0, 120.0]], dtype=float),
        botm=np.asarray([[[110.0, 110.0]], [[100.0, 100.0]], [[90.0, 90.0]]], dtype=float),
        materials=np.asarray([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=int),
        layer_names=("Q1_top", "Q1_mid", "Q1_base"),
        grid_corners_local=np.asarray([[0.0, 0.0], [20.0, 10.0]], dtype=float),
        grid_corners_global=np.asarray(
            [boundary_origin, boundary_origin + np.asarray([20.0, 10.0], dtype=float)],
            dtype=float,
        ),
        grid_corners_lonlat=np.asarray(
            [[14.0, 50.0], list(Transformer.from_crs(5514, 4326, always_xy=True).transform(origin_x + 20.0, origin_y + 10.0))],
            dtype=float,
        ),
    )


def _config_dict(wells_schema_path: Path, simulation_schema_path: Path) -> dict:
    return {
        "start_datetime": "2025-03-01T00:00:00",
        "model_3d": {
            "common": {
                "sim_name": "toy",
                "stress_periods_days": [1.0, 1.0],
            },
            "monitoring_wells": {
                "schema_path": str(wells_schema_path),
                "simulation_schema_path": str(simulation_schema_path),
                "well_ids": ["MW1"],
            },
        },
    }


def test_monitoring_well_loads_screened_layers_and_writes_predictions(tmp_path: Path) -> None:
    wells_schema_path = tmp_path / "wells_schema.yaml"
    wells_store_path = tmp_path / "wells.zarr"
    simulation_schema_path = tmp_path / "simulation_schema.yaml"
    simulation_store_path = tmp_path / "simulation.zarr"
    geometry = _geometry()
    well_lonlat = geometry.xy_local_to_lonlat(np.asarray([[5.0, 5.0]], dtype=float))[0]
    _build_wells_store(
        wells_schema_path,
        wells_store_path,
        longitude=float(well_lonlat[0]),
        latitude=float(well_lonlat[1]),
    )
    _write_simulation_schema(simulation_schema_path, simulation_store_path)

    config = _config_dict(wells_schema_path, simulation_schema_path)
    monitoring_wells = load_monitoring_wells(config_source=config, geometry=geometry)

    assert len(monitoring_wells) == 1
    assert monitoring_wells[0].row == 0
    assert monitoring_wells[0].col == 0
    assert monitoring_wells[0].layer_ids == (1, 2)

    heads_by_period = (
        np.asarray([[[118.0, 0.0]], [[108.0, 0.0]], [[98.0, 0.0]]], dtype=float),
        np.asarray([[[117.0, 0.0]], [[107.0, 0.0]], [[97.0, 0.0]]], dtype=float),
    )
    date_times = monitoring_output_times(
        config_source=config,
        stress_periods_days=(1.0, 1.0),
    )
    write_monitoring_well_predictions(
        config_source=config,
        monitoring_wells=monitoring_wells,
        date_times=date_times,
        heads_by_period=heads_by_period,
    )

    root = zarr_fuse.open_store(simulation_schema_path)
    dataset = root["Uhelna"]["well_prediction"].dataset

    assert dataset.sizes["date_time"] == 2
    assert dataset.sizes["well_id"] == 1
    assert dataset.sizes["calibration"] == 1
    assert np.allclose(np.asarray(dataset["water_level"]).reshape(-1), [103.0, 102.0])
    assert np.allclose(np.asarray(dataset["water_depth"]).reshape(-1), [17.0, 18.0])
