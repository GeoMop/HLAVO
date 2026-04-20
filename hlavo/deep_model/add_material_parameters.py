from __future__ import annotations

import argparse
import logging
from pathlib import Path

import attrs
import numpy as np

import hlavo.deep_model.model_3d_cfg as cfg3d
import hlavo.misc.config as cfg
from hlavo.deep_model.qgis_reader import GeometryConfig

LOG = logging.getLogger(__name__)

UNSAT_KEYS = (
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
    "ntrailwaves",
    "nwavesets",
    "simulate_et",
    "unsat_etwc",
    "unsat_etae",
    "simulate_gwseep",
    "hydraulic_conductivity",
    "vertical_conductivity",
    "specific_yield",
    "specific_storage",
    "initial_head_offset",
    "perlen",
    "nstp",
    "tsmult",
)


@attrs.define(frozen=True)
class MaterialAllSpec:
    horizontal_conductivity: float
    vertical_conductivity: float
    porosity: float
    vG_n: float
    vG_alpha: float
    unsat: dict[str, float | bool | int]


@attrs.define(frozen=True)
class LayerMaterialSpec:
    name: str
    horizontal_conductivity: float
    vertical_conductivity: float


@attrs.define(frozen=True)
class MaterialConfig:
    config_path: Path | None
    workspace_root: Path
    common: cfg3d.Model3DCommonConfig
    geometry: GeometryConfig
    output_path: Path
    defaults: MaterialAllSpec
    layers: tuple[LayerMaterialSpec, ...]

    @classmethod
    def from_source(
        cls,
        config_source: Path | dict,
        workspace: Path | None = None,
    ) -> "MaterialConfig":
        raw, config_path = cfg.load_config(config_source)
        common_raw = cfg3d.resolve_model_3d_common_raw(raw)
        common = cfg3d.Model3DCommonConfig.from_mapping(common_raw)
        geometry = GeometryConfig.from_source(config_source)
        materials_raw, _ = cfg.load_config(config_source, ("model_3d", "materials"))
        workspace_root = cfg3d.resolve_workspace_root(workspace, common_raw)
        output_path = Path(cfg3d.MATERIAL_PARAMETERS_FILENAME)
        all_raw = materials_raw["all"]
        assert isinstance(all_raw, dict), "materials.all must be a mapping"

        # TODO: simplify, distinct material vars and BC or other vars.
        unsat: dict[str, float | bool | int] = {}
        for key in UNSAT_KEYS:
            assert key in all_raw, f"materials.all.{key} is required"
            value = all_raw[key]
            if key in ("simulate_et", "unsat_etwc", "unsat_etae", "simulate_gwseep"):
                unsat[key] = bool(value)
            elif key in ("ntrailwaves", "nwavesets", "nstp"):
                unsat[key] = int(value)
            else:
                unsat[key] = float(value)

        layers: list[LayerMaterialSpec] = []
        for name, spec_raw in materials_raw.items():
            # TODO: these should not be under materials mapping
            if name in ("_config_path", "all", "material_parameters_output_path"):
                continue
            assert isinstance(spec_raw, dict), f"materials.{name} must be a mapping"
            layers.append(
                LayerMaterialSpec(
                    name=str(name),
                    horizontal_conductivity=float(spec_raw["horizontal_conductivity"]),
                    vertical_conductivity=float(spec_raw["vertical_conductivity"]),
                )
            )

        return cls(
            config_path=config_path,
            workspace_root=workspace_root,
            common=common,
            geometry=geometry,
            output_path=output_path,
            defaults=MaterialAllSpec(
                horizontal_conductivity=float(all_raw["horizontal_conductivity"]),
                vertical_conductivity=float(all_raw["vertical_conductivity"]),
                porosity=float(all_raw["porosity"]),
                vG_n=float(all_raw["vG_n"]),
                vG_alpha=float(all_raw["vG_alpha"]),
                unsat=unsat,
            ),
            layers=tuple(layers),
        )

    @property
    def workspace(self) -> Path:
        return cfg3d.resolve_model_workspace(self.workspace_root, self.common)

    @property
    def grid_path(self) -> Path:
        return self.geometry.resolve_grid_output_path(self.workspace)

    @property
    def material_parameters_path(self) -> Path:
        return cfg3d.resolve_model_relative_path(self.workspace, self.output_path)

    @property
    def layer_specs_by_name(self) -> dict[str, LayerMaterialSpec]:
        return {layer.name: layer for layer in self.layers}


def _layer_conductivities(
    layer_names: list[str],
    defaults: MaterialAllSpec,
    layer_specs: dict[str, LayerMaterialSpec],
) -> tuple[np.ndarray, np.ndarray]:
    kh = np.full(len(layer_names), defaults.horizontal_conductivity, dtype=float)
    kv = np.full(len(layer_names), defaults.vertical_conductivity, dtype=float)
    sand = layer_specs.get("sand")
    clay = layer_specs.get("clay")
    assert sand is not None, "materials.sand is required"
    assert clay is not None, "materials.clay is required"

    for idx, layer_name in enumerate(layer_names):
        if not layer_name.startswith("Q"):
            continue
        if layer_name.endswith("_base"):
            kh[idx] = sand.horizontal_conductivity
            kv[idx] = sand.vertical_conductivity
        elif layer_name.endswith("_top"):
            kh[idx] = clay.horizontal_conductivity
            kv[idx] = clay.vertical_conductivity

    return kh, kv


def write_material_model_files(
    config_source: Path | dict,
    workspace: Path | None = None,
) -> Path:
    mat_cfg = MaterialConfig.from_source(config_source, workspace=workspace)
    assert mat_cfg.grid_path.exists(), f"Grid NPZ not found: {mat_cfg.grid_path}"
    with np.load(mat_cfg.grid_path, allow_pickle=True) as data:
        grid_data = {key: data[key] for key in data.files}

    materials = np.asarray(grid_data["materials"], dtype=int)
    active_mask = np.asarray(grid_data["active_mask"], dtype=bool)
    top = np.asarray(grid_data["top"], dtype=float)
    botm = np.asarray(grid_data["botm"], dtype=float)
    layer_names = [str(name) for name in grid_data["layer_names"].tolist()]

    assert materials.ndim == 3, "materials must be 3D"
    nz, ny, nx = materials.shape
    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"
    assert top.shape == (ny, nx), "top shape mismatch"
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"

    idomain = np.broadcast_to(active_mask, (nz, ny, nx)).astype(int)
    materials_mf = materials[::-1, :, :]

    kh_values, kv_values = _layer_conductivities(
        layer_names,
        mat_cfg.defaults,
        mat_cfg.layer_specs_by_name,
    )

    valid = materials_mf >= 0
    kh = np.full((nz, ny, nx), mat_cfg.defaults.horizontal_conductivity, dtype=float)
    kv = np.full((nz, ny, nx), mat_cfg.defaults.vertical_conductivity, dtype=float)
    porosity = np.full((nz, ny, nx), mat_cfg.defaults.porosity, dtype=float)
    vg_alpha = np.full((nz, ny, nx), mat_cfg.defaults.vG_alpha, dtype=float)
    vg_n = np.full((nz, ny, nx), mat_cfg.defaults.vG_n, dtype=float)

    kh[valid] = kh_values[materials_mf[valid]]
    kv[valid] = kv_values[materials_mf[valid]]

    mat_cfg.material_parameters_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "top": top,
        "botm": botm,
        "materials": materials,
        "materials_mf": materials_mf,
        "active_mask": active_mask,
        "idomain": idomain,
        "kh": kh,
        "kv": kv,
        "hk": kh,
        "k33": kv,
        "porosity": porosity,
        "van_genuchten_alpha": vg_alpha,
        "van_genuchten_n": vg_n,
        "layer_names": np.asarray(layer_names, dtype=object),
    }
    for key, value in mat_cfg.defaults.unsat.items():
        payload[key] = np.asarray(value)

    np.savez_compressed(mat_cfg.material_parameters_path, **payload)
    assert mat_cfg.material_parameters_path.exists(), (
        f"Failed to write material parameter file: {mat_cfg.material_parameters_path}"
    )
    LOG.info("Saved material parameters to %s", mat_cfg.material_parameters_path)
    return mat_cfg.material_parameters_path


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Add per-material arrays from config materials definitions."
#     )
#     parser.add_argument(
#         "--config",
#         type=Path,
#         default=Path("model_config.yaml"),
#         help="Path to model_config.yaml",
#     )
#     args = parser.parse_args()
#
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
#     write_material_model_files(args.config)
#
#
# if __name__ == "__main__":
#     main()
