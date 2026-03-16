from __future__ import annotations

import argparse
import logging
from pathlib import Path

import attrs
import numpy as np
import yaml

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
    layers: tuple[str, ...]
    horizontal_conductivity: float
    vertical_conductivity: float


@attrs.define(frozen=True)
class MaterialConfig:
    config_path: Path
    grid_path: Path
    output_path: Path
    defaults: MaterialAllSpec
    layer_materials: dict[str, LayerMaterialSpec]

    @staticmethod
    def from_yaml(config_path: Path) -> "MaterialConfig":
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict), "Config YAML must be a mapping"

        model_raw = raw.get("model", {})
        assert isinstance(model_raw, dict), "model config must be a mapping"
        model_name = str(model_raw["model_name"])
        workspace = Path(str(model_raw.get("workspace", "model")))
        run_workspace = workspace / model_name

        grid_raw = raw.get("grid_output_path", "grid_materials.npz")
        grid_path = Path(str(grid_raw))
        if not grid_path.is_absolute():
            grid_path = run_workspace / grid_path

        output_raw = raw.get("material_parameters_output_path", "material_parameters.npz")
        output_path = Path(str(output_raw))
        if not output_path.is_absolute():
            output_path = run_workspace / output_path

        materials_raw = raw.get("materials", {})
        assert isinstance(materials_raw, dict), "materials must be a mapping"
        assert "all" in materials_raw, "materials must include 'all' defaults"
        all_raw = materials_raw["all"]
        assert isinstance(all_raw, dict), "materials.all must be a mapping"

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

        defaults = MaterialAllSpec(
            horizontal_conductivity=float(all_raw["horizontal_conductivity"]),
            vertical_conductivity=float(all_raw["vertical_conductivity"]),
            porosity=float(all_raw["porosity"]),
            vG_n=float(all_raw["vG_n"]),
            vG_alpha=float(all_raw["vG_alpha"]),
            unsat=unsat,
        )

        layer_materials: dict[str, LayerMaterialSpec] = {}
        for material_name, material_raw in materials_raw.items():
            if material_name == "all":
                continue
            assert isinstance(material_raw, dict), f"materials.{material_name} must be a mapping"
            layers_raw = material_raw.get("layers", [])
            assert isinstance(layers_raw, list), f"materials.{material_name}.layers must be a list"
            layer_materials[str(material_name)] = LayerMaterialSpec(
                name=str(material_name),
                layers=tuple(str(layer) for layer in layers_raw),
                horizontal_conductivity=float(material_raw["horizontal_conductivity"]),
                vertical_conductivity=float(material_raw["vertical_conductivity"]),
            )

        return MaterialConfig(
            config_path=config_path,
            grid_path=grid_path,
            output_path=output_path,
            defaults=defaults,
            layer_materials=layer_materials,
        )


def _layer_conductivities(
    layer_names: list[str],
    defaults: MaterialAllSpec,
    layer_materials: dict[str, LayerMaterialSpec],
) -> tuple[np.ndarray, np.ndarray]:
    kh = np.full(len(layer_names), defaults.horizontal_conductivity, dtype=float)
    kv = np.full(len(layer_names), defaults.vertical_conductivity, dtype=float)

    layer_to_idx = {name: idx for idx, name in enumerate(layer_names)}
    assigned = set()
    for material_name, material in layer_materials.items():
        for layer in material.layers:
            assert layer in layer_to_idx, f"Unknown layer {layer} in materials.{material_name}.layers"
            assert layer not in assigned, f"Layer {layer} assigned by multiple materials"
            idx = layer_to_idx[layer]
            kh[idx] = material.horizontal_conductivity
            kv[idx] = material.vertical_conductivity
            assigned.add(layer)

    return kh, kv


def add_material_parameters(config_path: Path) -> Path:
    cfg = MaterialConfig.from_yaml(config_path)
    assert cfg.grid_path.exists(), f"Grid NPZ not found: {cfg.grid_path}"
    with np.load(cfg.grid_path, allow_pickle=True) as data:
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

    kh_values, kv_values = _layer_conductivities(layer_names, cfg.defaults, cfg.layer_materials)

    valid = materials_mf >= 0
    kh = np.full((nz, ny, nx), cfg.defaults.horizontal_conductivity, dtype=float)
    kv = np.full((nz, ny, nx), cfg.defaults.vertical_conductivity, dtype=float)
    porosity = np.full((nz, ny, nx), cfg.defaults.porosity, dtype=float)
    vg_alpha = np.full((nz, ny, nx), cfg.defaults.vG_alpha, dtype=float)
    vg_n = np.full((nz, ny, nx), cfg.defaults.vG_n, dtype=float)

    kh[valid] = kh_values[materials_mf[valid]]
    kv[valid] = kv_values[materials_mf[valid]]

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
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
    for key, value in cfg.defaults.unsat.items():
        payload[key] = np.asarray(value)

    np.savez_compressed(cfg.output_path, **payload)
    assert cfg.output_path.exists(), f"Failed to write material parameter file: {cfg.output_path}"
    LOG.info("Saved material parameters to %s", cfg.output_path)
    return cfg.output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add per-material arrays from config materials definitions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to model_config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    add_material_parameters(args.config)


if __name__ == "__main__":
    main()
