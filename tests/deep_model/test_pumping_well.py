from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import zarr_fuse
from pyproj import Transformer

from hlavo.deep_model.add_material_parameters import material_dataset_from_config
from hlavo.deep_model.model_3d_cfg import Model3DCommonConfig
from hlavo.deep_model.pumping_well import load_pumping_wells
from hlavo.deep_model.qgis_reader import Grid, ModelGeometry
from hlavo.deep_model.simulation_builder import build_modflow_simulation


def _write_wells_schema(schema_path: Path, store_path: Path) -> None:
    schema_path.write_text(
        f"""
ATTRS:
  STORE_URL: "{store_path}"
  LOGGER: "local"
Uhelna:
  water_draw:
    VARS:
      cum_draw:
        type: float
        coords: ["date", "well_id"]
      longitude:
        type: float
        coords: ["well_id"]
      latitude:
        type: float
        coords: ["well_id"]
    COORDS:
      date:
        unit: {{ tick: "h", tz: "UTC" }}
      well_id:
        type: str[16]
        sorted: []
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
    root["Uhelna"]["water_draw"].update(
        pl.DataFrame(
            [
                {
                    "date": "2025-03-01T00:00",
                    "well_id": "P1",
                    "cum_draw": 100.0,
                    "longitude": longitude,
                    "latitude": latitude,
                },
                {
                    "date": "2025-03-02T00:00",
                    "well_id": "P1",
                    "cum_draw": 50.0,
                    "longitude": longitude,
                    "latitude": latitude,
                },
            ]
        )
    )


def _config_dict(schema_path: Path) -> dict:
    return {
        "model_3d": {
            "common": {
                "sim_name": "toy",
                "exe_name": "mf6",
                "drain_conductance": 0.25,
                "simulation_days": 2.0,
                "stress_periods_days": [1.0, 1.0],
            },
            "materials": {
                "all": {
                    "horizontal_conductivity": [1.0e-7, 1.0e-6, 1.0e-5],
                    "vertical_conductivity": [1.0e-8, 1.0e-7, 1.0e-6],
                    "porosity": [0.2, 0.3, 0.4],
                    "vG_n": [1.5, 2.0, 2.5],
                    "vG_alpha": [0.5, 1.0, 1.5],
                    "recharge_rate": [0.0, 1.0e-4, 2.0e-4],
                    "vks": [1.0e-7, 1.0e-6, 1.0e-5],
                    "thtr": [0.01, 0.05, 0.1],
                    "thts": [0.2, 0.3, 0.4],
                    "thti": [0.15, 0.2, 0.25],
                    "eps": [2.5, 3.5, 4.5],
                    "surfdep": [0.0, 0.0, 0.0],
                    "pet": [0.0, 0.0, 0.0],
                    "extdp": [0.5, 1.0, 1.5],
                    "extwc": [0.05, 0.1, 0.2],
                    "ha": [0.0, 0.0, 0.0],
                    "hroot": [0.0, 0.0, 0.0],
                    "rootact": [0.0, 0.0, 0.0],
                    "hydraulic_conductivity": [1.0e-7, 1.0e-6, 1.0e-5],
                    "specific_yield": [0.05, 0.1, 0.2],
                    "specific_storage": [1.0e-6, 1.0e-5, 1.0e-4],
                    "initial_head_offset": [-2.0, -1.0, 0.0],
                    "perlen": [1.0, 1.0, 1.0],
                    "tsmult": [1.0, 1.0, 1.0],
                    "ntrailwaves": 7,
                    "nwavesets": 40,
                    "nstp": 1,
                    "simulate_et": False,
                    "unsat_etwc": False,
                    "unsat_etae": False,
                    "simulate_gwseep": True,
                },
                "sand": {
                    "horizontal_conductivity": [1.0e-6, 1.0e-5, 1.0e-4],
                    "vertical_conductivity": [5.0e-7, 5.0e-6, 5.0e-5],
                },
                "clay": {
                    "horizontal_conductivity": [1.0e-9, 1.0e-8, 1.0e-7],
                    "vertical_conductivity": [5.0e-10, 5.0e-9, 5.0e-8],
                },
            },
            "pumping_wells": {
                "schema_path": str(schema_path),
                "well_ids": ["P1"],
            },
        }
    }


def _geometry() -> ModelGeometry:
    to_jtsk = Transformer.from_crs(4326, 5514, always_xy=True)
    origin_x, origin_y = to_jtsk.transform(14.0, 50.0)
    boundary_origin = np.asarray([origin_x, origin_y], dtype=float)
    return ModelGeometry(
        boundary_origin=boundary_origin,
        grid=Grid(
            origin=np.asarray([0.0, 0.0, -10.0], dtype=float),
            step=np.asarray([10.0, 10.0, 5.0], dtype=float),
            el_dims=np.asarray([2, 1, 2], dtype=int),
        ),
        rasters=(),
        active_mask=np.asarray([[True, True]], dtype=bool),
        top=np.asarray([[0.0, 0.0]], dtype=float),
        botm=np.asarray([[[-5.0, -5.0]], [[-10.0, -10.0]]], dtype=float),
        materials=np.asarray([[[0, 1]], [[0, 1]]], dtype=int),
        layer_names=("Q1_top", "Q1_base"),
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


def test_pumping_well_builds_wel_package(tmp_path: Path) -> None:
    schema_path = tmp_path / "wells_schema.yaml"
    store_path = tmp_path / "wells.zarr"
    geometry = _geometry()
    well_lonlat = geometry.xy_local_to_lonlat(np.asarray([[5.0, 5.0]], dtype=float))[0]
    _build_wells_store(
        schema_path,
        store_path,
        longitude=float(well_lonlat[0]),
        latitude=float(well_lonlat[1]),
    )

    config = _config_dict(schema_path)
    common = Model3DCommonConfig.from_mapping(config["model_3d"]["common"])
    material_dataset = material_dataset_from_config(config)
    pumping_wells = load_pumping_wells(
        config_source=config,
        common=common,
        geometry=geometry,
    )

    assert len(pumping_wells) == 1
    assert pumping_wells[0].rates_m3_per_day == (-100.0, -50.0)

    workdir = tmp_path / "simulation"
    build_modflow_simulation(
        common=common,
        geometry=geometry,
        material_dataset=material_dataset,
        workspace=workdir,
        pumping_wells=pumping_wells,
        exe_name="mf6",
    )

    wel_text = (workdir / "toy.wel").read_text(encoding="utf-8")
    assert "BEGIN period  1" in wel_text
    assert "BEGIN period  2" in wel_text
    assert "-1.00000000E+02" in wel_text
    assert "-5.00000000E+01" in wel_text
