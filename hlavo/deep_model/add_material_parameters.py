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
    "vertical_conductivity",
    "specific_yield",
    "specific_storage",
    "initial_head_offset",
)


@attrs.define(frozen=True)
class MaterialParameterSet:
    horizontal_conductivity: float
    vertical_conductivity: float
    porosity: float
    vG_n: float
    vG_alpha: float


@attrs.define(frozen=True)
class MaterialAllSpec:
    params: MaterialParameterSet
    unsat: dict[str, float | bool | int]


@attrs.define(frozen=True)
class GeologicalLayerSpec:
    name: str
    top_interface: str
    bottom_interface: str
    set_name: str | None
    override: MaterialParameterSet | None


@attrs.define(frozen=True)
class MaterialConfig:
    config_path: Path
    grid_path: Path
    output_path: Path
    defaults: MaterialAllSpec
    sets: dict[str, MaterialParameterSet]
    geological_layers: tuple[GeologicalLayerSpec, ...]
    strict_layer_mapping: bool

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

        defaults = MaterialAllSpec(
            params=MaterialParameterSet(
                horizontal_conductivity=float(all_raw["horizontal_conductivity"]),
                vertical_conductivity=float(all_raw["vertical_conductivity"]),
                porosity=float(all_raw["porosity"]),
                vG_n=float(all_raw["vG_n"]),
                vG_alpha=float(all_raw["vG_alpha"]),
            ),
            unsat=_read_unsat(all_raw),
        )

        sets_raw = materials_raw.get("sets", {})
        assert isinstance(sets_raw, dict), "materials.sets must be a mapping"
        sets: dict[str, MaterialParameterSet] = {}
        for set_name, set_raw in sets_raw.items():
            assert isinstance(set_raw, dict), f"materials.sets.{set_name} must be a mapping"
            sets[str(set_name)] = MaterialParameterSet(
                horizontal_conductivity=float(set_raw["horizontal_conductivity"]),
                vertical_conductivity=float(set_raw["vertical_conductivity"]),
                porosity=float(set_raw.get("porosity", defaults.params.porosity)),
                vG_n=float(set_raw.get("vG_n", defaults.params.vG_n)),
                vG_alpha=float(set_raw.get("vG_alpha", defaults.params.vG_alpha)),
            )

        layers_raw = materials_raw.get("layers", [])
        assert isinstance(layers_raw, list), "materials.layers must be a list"
        geological_layers: list[GeologicalLayerSpec] = []
        for i, layer_raw in enumerate(layers_raw):
            assert isinstance(layer_raw, dict), f"materials.layers[{i}] must be a mapping"
            set_name_raw = layer_raw.get("set")
            set_name = str(set_name_raw) if set_name_raw is not None else None
            if set_name is not None:
                assert set_name in sets, f"Unknown set '{set_name}' in materials.layers[{i}]"

            override = None
            if any(k in layer_raw for k in ("horizontal_conductivity", "vertical_conductivity", "porosity", "vG_n", "vG_alpha")):
                override = MaterialParameterSet(
                    horizontal_conductivity=float(
                        layer_raw.get(
                            "horizontal_conductivity",
                            sets[set_name].horizontal_conductivity if set_name else defaults.params.horizontal_conductivity,
                        )
                    ),
                    vertical_conductivity=float(
                        layer_raw.get(
                            "vertical_conductivity",
                            sets[set_name].vertical_conductivity if set_name else defaults.params.vertical_conductivity,
                        )
                    ),
                    porosity=float(
                        layer_raw.get(
                            "porosity",
                            sets[set_name].porosity if set_name else defaults.params.porosity,
                        )
                    ),
                    vG_n=float(layer_raw.get("vG_n", sets[set_name].vG_n if set_name else defaults.params.vG_n)),
                    vG_alpha=float(
                        layer_raw.get("vG_alpha", sets[set_name].vG_alpha if set_name else defaults.params.vG_alpha)
                    ),
                )

            geological_layers.append(
                GeologicalLayerSpec(
                    name=str(layer_raw.get("name", f"layer_{i}")),
                    top_interface=str(layer_raw["top_interface"]),
                    bottom_interface=str(layer_raw["bottom_interface"]),
                    set_name=set_name,
                    override=override,
                )
            )

        strict_layer_mapping = bool(materials_raw.get("strict_layer_mapping", False))

        return MaterialConfig(
            config_path=config_path,
            grid_path=grid_path,
            output_path=output_path,
            defaults=defaults,
            sets=sets,
            geological_layers=tuple(geological_layers),
            strict_layer_mapping=strict_layer_mapping,
        )


def _read_unsat(all_raw: dict) -> dict[str, float | bool | int]:
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
    return unsat


def _resolve_layer_parameters(
    spec: GeologicalLayerSpec,
    defaults: MaterialParameterSet,
    sets: dict[str, MaterialParameterSet],
) -> MaterialParameterSet:
    if spec.override is not None:
        return spec.override
    if spec.set_name is not None:
        return sets[spec.set_name]
    return defaults


def _layer_parameters_from_geology(
    layer_names: list[str],
    defaults: MaterialParameterSet,
    sets: dict[str, MaterialParameterSet],
    geological_layers: tuple[GeologicalLayerSpec, ...],
    strict_layer_mapping: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n_layers = len(layer_names)
    kh = np.full(n_layers, defaults.horizontal_conductivity, dtype=float)
    kv = np.full(n_layers, defaults.vertical_conductivity, dtype=float)
    porosity = np.full(n_layers, defaults.porosity, dtype=float)
    vg_n = np.full(n_layers, defaults.vG_n, dtype=float)
    vg_alpha = np.full(n_layers, defaults.vG_alpha, dtype=float)
    material_class = np.zeros(n_layers, dtype=np.int16)  # 0=other, 1=sand, 2=clay
    material_names = [str(name) for name in layer_names]

    interface_to_idx = {name: idx for idx, name in enumerate(layer_names)}  # bottom-up indices
    topdown = list(reversed(layer_names))
    topdown_to_idx = {name: idx for idx, name in enumerate(topdown)}

    for spec in geological_layers:
        top_exists = spec.top_interface in topdown_to_idx
        bottom_exists = spec.bottom_interface in topdown_to_idx
        if not (top_exists and bottom_exists):
            message = (
                f"Layer mapping '{spec.name}' references missing interface(s): "
                f"{spec.top_interface}, {spec.bottom_interface}"
            )
            if strict_layer_mapping:
                raise AssertionError(message)
            LOG.warning(message)
            continue

        top_i = topdown_to_idx[spec.top_interface]
        bot_i = topdown_to_idx[spec.bottom_interface]
        if bot_i != top_i + 1:
            message = (
                f"Layer mapping '{spec.name}' interfaces are not consecutive in top-down order: "
                f"{spec.top_interface} -> {spec.bottom_interface}"
            )
            if strict_layer_mapping:
                raise AssertionError(message)
            LOG.warning(message)
            continue

        bottom_interface_idx = interface_to_idx[spec.bottom_interface]
        params = _resolve_layer_parameters(spec, defaults, sets)
        kh[bottom_interface_idx] = params.horizontal_conductivity
        kv[bottom_interface_idx] = params.vertical_conductivity
        porosity[bottom_interface_idx] = params.porosity
        vg_n[bottom_interface_idx] = params.vG_n
        vg_alpha[bottom_interface_idx] = params.vG_alpha
        material_class[bottom_interface_idx] = _material_class_from_spec(spec)
        material_names[bottom_interface_idx] = (
            spec.name.strip() if str(spec.name).strip() else spec.bottom_interface
        )

    return kh, kv, porosity, vg_n, vg_alpha, material_class, material_names


def _material_class_from_spec(spec: GeologicalLayerSpec) -> np.int16:
    haystack = " ".join(
        value.lower() for value in (spec.name, spec.set_name or "") if value
    )
    if "sand" in haystack:
        return np.int16(1)
    if "clay" in haystack:
        return np.int16(2)
    return np.int16(0)


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
    # Keep legacy export key, but properties/classes are mapped from the native
    # grid orientation (same as top/botm arrays and MODFLOW layer order).
    materials_mf = materials[::-1, :, :]

    (
        kh_values,
        kv_values,
        porosity_values,
        vg_n_values,
        vg_alpha_values,
        material_class_values,
        material_names,
    ) = _layer_parameters_from_geology(
        layer_names,
        cfg.defaults.params,
        cfg.sets,
        cfg.geological_layers,
        cfg.strict_layer_mapping,
    )

    valid = materials >= 0
    kh = np.full((nz, ny, nx), cfg.defaults.params.horizontal_conductivity, dtype=float)
    kv = np.full((nz, ny, nx), cfg.defaults.params.vertical_conductivity, dtype=float)
    porosity = np.full((nz, ny, nx), cfg.defaults.params.porosity, dtype=float)
    vg_alpha = np.full((nz, ny, nx), cfg.defaults.params.vG_alpha, dtype=float)
    vg_n = np.full((nz, ny, nx), cfg.defaults.params.vG_n, dtype=float)
    material_class = np.full((nz, ny, nx), -1, dtype=np.int16)

    kh[valid] = kh_values[materials[valid]]
    kv[valid] = kv_values[materials[valid]]
    porosity[valid] = porosity_values[materials[valid]]
    vg_alpha[valid] = vg_alpha_values[materials[valid]]
    vg_n[valid] = vg_n_values[materials[valid]]
    material_class[valid] = material_class_values[materials[valid]]

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
        "material_class": material_class,
        "material_class_by_layer": material_class_values,
        "material_name_by_layer": np.asarray(material_names, dtype=object),
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
