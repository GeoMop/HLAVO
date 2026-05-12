from __future__ import annotations

import logging
from pathlib import Path

import attrs
import numpy as np
import xarray as xr

import hlavo.deep_model.model_3d_cfg as cfg3d
import hlavo.misc.config as cfg
from hlavo.deep_model.qgis_reader import GeometryConfig, ModelGeometry
from hlavo.deep_model.simulation_builder import build_material_fields

LOG = logging.getLogger(__name__)

BOUND_NAMES = ("lo", "init", "hi")
BOUNDED_FLOAT_KEYS = (
    "horizontal_conductivity",
    "vertical_conductivity",
    "porosity",
    "vG_n",
    "vG_alpha",
    "recharge_rate",
    "vks",
    "thtr",
    "thts",
    "thti",
    "eps",
    "surfdep",
    "pet",
    "extdp",
    "extwc",
    "ha",
    "hroot",
    "rootact",
    "hydraulic_conductivity",
    "specific_yield",
    "specific_storage",
    "initial_head_offset",
    "perlen",
    "tsmult",
)
INT_KEYS = ("ntrailwaves", "nwavesets", "nstp")
BOOL_KEYS = ("simulate_et", "unsat_etwc", "unsat_etae", "simulate_gwseep")
MATERIAL_DATASET_KEYS = BOUNDED_FLOAT_KEYS + INT_KEYS + BOOL_KEYS


@attrs.define(frozen=True)
class MaterialConfig:
    config_path: Path | None
    workspace_root: Path
    common: cfg3d.Model3DCommonConfig
    geometry: GeometryConfig | None
    output_path: Path
    raw_materials: dict

    @classmethod
    def from_source(
        cls,
        config_source: Path | dict,
        workspace: Path | None = None,
    ) -> "MaterialConfig":
        raw, config_path = cfg.load_config(config_source)
        common_raw = cfg3d.resolve_model_3d_common_raw(raw)
        common = cfg3d.Model3DCommonConfig.from_mapping(common_raw)
        geometry = None
        model_3d_raw = cfg3d.resolve_model_3d_section(raw)
        if "geometry" in model_3d_raw or "qgis_project_path" in raw:
            geometry = GeometryConfig.from_source(config_source)
        if "materials" in model_3d_raw:
            materials_raw = model_3d_raw["materials"]
        else:
            materials_raw = raw["materials"]
        assert isinstance(materials_raw, dict), "materials must be a mapping"
        assert "all" in materials_raw, "materials must include the virtual 'all' material"
        workspace_root = cfg3d.resolve_workspace_root(workspace, common_raw)
        return cls(
            config_path=config_path,
            workspace_root=workspace_root,
            common=common,
            geometry=geometry,
            output_path=Path(cfg3d.MATERIAL_PARAMETERS_FILENAME),
            raw_materials=materials_raw,
        )

    @property
    def workspace(self) -> Path:
        return cfg3d.resolve_model_workspace(self.workspace_root, self.common)

    @property
    def grid_path(self) -> Path:
        assert self.geometry is not None, "Geometry config is required to resolve the grid path"
        return self.geometry.resolve_grid_output_path(self.workspace)

    @property
    def material_parameters_path(self) -> Path:
        return cfg3d.resolve_model_relative_path(self.workspace, self.output_path)


def _material_names(materials_raw: dict) -> tuple[str, ...]:
    names = [str(name) for name in materials_raw if name not in ("_config_path",)]
    assert "all" in names, "materials must include the virtual 'all' material"
    ordered = ["all"] + sorted(name for name in names if name != "all")
    return tuple(ordered)


def _bounded_triplet(raw_value: object, key: str) -> tuple[float, float, float]:
    if isinstance(raw_value, (list, tuple)):
        assert len(raw_value) == 3, f"materials.*.{key} must be scalar or length-3 [lo, init, hi]"
        lo, init, hi = (float(value) for value in raw_value)
    else:
        lo = init = hi = float(raw_value)
    assert lo <= init <= hi, f"materials.*.{key} must satisfy lo <= init <= hi"
    return (lo, init, hi)


def material_dataset_from_config(
    config_source: Path | dict,
    workspace: Path | None = None,
) -> xr.Dataset:
    mat_cfg = MaterialConfig.from_source(config_source, workspace=workspace)
    materials_raw = mat_cfg.raw_materials
    material_names = _material_names(materials_raw)
    defaults_raw = materials_raw["all"]
    assert isinstance(defaults_raw, dict), "materials.all must be a mapping"

    coords = {
        "material": np.asarray(material_names, dtype=object),
        "bound": np.asarray(BOUND_NAMES, dtype=object),
    }
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    for key in BOUNDED_FLOAT_KEYS:
        rows = []
        for material_name in material_names:
            material_raw = materials_raw[material_name]
            assert isinstance(material_raw, dict), f"materials.{material_name} must be a mapping"
            source_value = material_raw[key] if key in material_raw else defaults_raw[key]
            rows.append(_bounded_triplet(source_value, key))
        data_vars[key] = (("material", "bound"), np.asarray(rows, dtype=float))

    for key in INT_KEYS:
        values = []
        for material_name in material_names:
            material_raw = materials_raw[material_name]
            source_value = material_raw[key] if key in material_raw else defaults_raw[key]
            values.append(int(source_value))
        data_vars[key] = (("material",), np.asarray(values, dtype=int))

    for key in BOOL_KEYS:
        values = []
        for material_name in material_names:
            material_raw = materials_raw[material_name]
            source_value = material_raw[key] if key in material_raw else defaults_raw[key]
            values.append(bool(source_value))
        data_vars[key] = (("material",), np.asarray(values, dtype=bool))

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset.attrs["virtual_default_material"] = "all"
    dataset.attrs["config_path"] = "" if mat_cfg.config_path is None else str(mat_cfg.config_path)
    return dataset


def write_material_model_files(
    config_source: Path | dict,
    workspace: Path | None = None,
) -> Path:
    mat_cfg = MaterialConfig.from_source(config_source, workspace=workspace)
    geometry = ModelGeometry.from_npz(mat_cfg.grid_path)
    material_dataset = material_dataset_from_config(config_source, workspace=workspace)
    fields = build_material_fields(geometry=geometry, material_dataset=material_dataset)

    mat_cfg.material_parameters_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "top": fields.top,
        "botm": fields.botm,
        "materials": fields.materials,
        "materials_mf": fields.materials_mf,
        "active_mask": fields.active_mask,
        "idomain": fields.idomain,
        "kh": fields.kh,
        "kv": fields.kv,
        "hk": fields.kh,
        "k33": fields.kv,
        "porosity": fields.porosity,
        "van_genuchten_alpha": fields.vg_alpha,
        "van_genuchten_n": fields.vg_n,
        "layer_names": np.asarray(fields.layer_names, dtype=object),
        "material_names": np.asarray(material_dataset.coords["material"].values.tolist(), dtype=object),
        "bound_names": np.asarray(material_dataset.coords["bound"].values.tolist(), dtype=object),
    }
    for key in MATERIAL_DATASET_KEYS:
        values = material_dataset[key].values
        payload[key] = np.asarray(values)

    np.savez_compressed(mat_cfg.material_parameters_path, **payload)
    assert mat_cfg.material_parameters_path.exists(), (
        f"Failed to write material parameter file: {mat_cfg.material_parameters_path}"
    )
    LOG.info("Saved material parameters to %s", mat_cfg.material_parameters_path)
    return mat_cfg.material_parameters_path
