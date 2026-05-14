from __future__ import annotations

from pathlib import Path

import numpy as np

from hlavo.deep_model.add_material_parameters import material_dataset_from_config
from hlavo.deep_model.model_3d_cfg import Model3DCommonConfig
from hlavo.deep_model.qgis_reader import Grid, ModelGeometry
from hlavo.deep_model.simulation_builder import build_material_fields, build_modflow_simulation


def _config_dict() -> dict:
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
        }
    }


def _geometry() -> ModelGeometry:
    return ModelGeometry(
        boundary_origin=np.asarray([0.0, 0.0], dtype=float),
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
        grid_corners_global=np.asarray([[0.0, 0.0], [20.0, 10.0]], dtype=float),
        grid_corners_lonlat=np.asarray([[14.0, 50.0], [14.001, 50.001]], dtype=float),
    )


def test_material_dataset_from_config_builds_bounded_xarray() -> None:
    dataset = material_dataset_from_config(_config_dict())

    assert tuple(dataset.coords["material"].values.tolist()) == ("all", "clay", "sand")
    assert tuple(dataset.coords["bound"].values.tolist()) == ("lo", "init", "hi")
    assert float(dataset["horizontal_conductivity"].sel(material="all", bound="init")) == 1.0e-6
    assert float(dataset["horizontal_conductivity"].sel(material="sand", bound="init")) == 1.0e-5
    assert bool(dataset["simulate_gwseep"].sel(material="all")) is True


def test_build_modflow_simulation_uses_shared_material_fields(tmp_path: Path) -> None:
    config = _config_dict()
    geometry = _geometry()
    material_dataset = material_dataset_from_config(config)
    common = Model3DCommonConfig.from_mapping(config["model_3d"]["common"])

    fields = build_material_fields(geometry=geometry, material_dataset=material_dataset)
    assert np.allclose(fields.kh[:, 0, 0], 1.0e-8)
    assert np.allclose(fields.kh[:, 0, 1], 1.0e-5)
    assert fields.specific_yield == 0.1

    workdir = tmp_path / "simulation"
    result = build_modflow_simulation(
        common=common,
        geometry=geometry,
        material_dataset=material_dataset,
        workspace=workdir,
        exe_name="mf6",
    )

    assert (workdir / "mfsim.nam").exists()
    assert (workdir / "toy.nam").exists()
    assert (workdir / "toy.dis").exists()
    assert (workdir / "toy.npf").exists()
    assert result.material_fields.idomain.shape == (2, 1, 2)
