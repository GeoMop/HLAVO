from __future__ import annotations

import argparse
import logging
from pathlib import Path

import attrs
import numpy as np
import yaml

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class MockConfig:
    config_path: Path
    workspace: Path
    model_name: str
    grid_output_path: Path
    material_output_path: Path
    sim_name: str
    nx: int
    ny: int
    nz: int
    delr: float
    delc: float
    delz: np.ndarray
    top: float
    layer_names: tuple[str, ...]
    material_by_layer: np.ndarray
    material_class_by_layer: np.ndarray
    hk_by_layer: np.ndarray
    kv_by_layer: np.ndarray
    porosity_by_layer: np.ndarray
    vg_alpha_by_layer: np.ndarray
    vg_n_by_layer: np.ndarray
    material_name_by_layer: tuple[str, ...]
    recharge_rate: float
    specific_yield: float
    specific_storage: float

    @staticmethod
    def from_yaml(config_path: Path) -> "MockConfig":
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict), "Config YAML must be a mapping"

        model_raw = raw.get("model", {})
        assert isinstance(model_raw, dict), "model must be a mapping"
        model_name = str(model_raw.get("model_name", "mock_cube"))
        sim_name = str(model_raw.get("sim_name", "mock_cube"))
        workspace = Path(str(model_raw.get("workspace", "model"))) / model_name

        grid_output_path = Path(str(raw.get("grid_output_path", "grid_materials.npz")))
        if not grid_output_path.is_absolute():
            grid_output_path = workspace / grid_output_path
        material_output_path = Path(
            str(raw.get("material_parameters_output_path", "material_parameters.npz"))
        )
        if not material_output_path.is_absolute():
            material_output_path = workspace / material_output_path

        mock_raw = raw.get("mock", {})
        assert isinstance(mock_raw, dict), "mock section is required"
        nx = int(mock_raw.get("nx", 8))
        ny = int(mock_raw.get("ny", 8))
        nz = int(mock_raw.get("nz", 6))
        assert nx > 0 and ny > 0 and nz > 0, "mock nx/ny/nz must be > 0"

        delr = float(mock_raw.get("delr", 10.0))
        delc = float(mock_raw.get("delc", 10.0))
        assert delr > 0 and delc > 0, "mock delr/delc must be > 0"

        delz_raw = mock_raw.get("delz", 1.0)
        if isinstance(delz_raw, list):
            delz = np.asarray([float(v) for v in delz_raw], dtype=float)
            assert delz.size == nz, "mock.delz list length must equal mock.nz"
        else:
            delz = np.full(nz, float(delz_raw), dtype=float)
        assert np.all(delz > 0), "mock.delz must be > 0"

        top = float(mock_raw.get("top", 100.0))

        default_layers = tuple(f"layer_{i+1:02d}" for i in range(nz))
        layer_names_raw = mock_raw.get("layer_names", list(default_layers))
        assert isinstance(layer_names_raw, list), "mock.layer_names must be a list"
        assert len(layer_names_raw) == nz, "mock.layer_names length must equal mock.nz"
        layer_names = tuple(str(v) for v in layer_names_raw)

        material_by_layer_raw = mock_raw.get("material_by_layer", list(range(nz)))
        assert isinstance(material_by_layer_raw, list), "mock.material_by_layer must be a list"
        assert len(material_by_layer_raw) == nz, "mock.material_by_layer length must equal mock.nz"
        material_by_layer = np.asarray([int(v) for v in material_by_layer_raw], dtype=int)
        assert np.all(material_by_layer >= 0), "mock.material_by_layer values must be >= 0"
        n_material_ids = int(material_by_layer.max()) + 1

        def _as_layer_array(key: str, default: float) -> np.ndarray:
            value = mock_raw.get(key, default)
            if isinstance(value, list):
                arr = np.asarray([float(v) for v in value], dtype=float)
                assert arr.size == nz, f"mock.{key} length must equal mock.nz"
                return arr
            return np.full(nz, float(value), dtype=float)

        def _as_material_names() -> tuple[str, ...]:
            names_raw = mock_raw.get("material_name_by_layer")
            if names_raw is not None:
                assert isinstance(names_raw, list), "mock.material_name_by_layer must be a list"
                assert len(names_raw) == nz, "mock.material_name_by_layer length must equal mock.nz"
                return tuple(str(v) for v in names_raw)
            return layer_names

        material_class_raw = mock_raw.get("material_class_by_layer", 0)
        if isinstance(material_class_raw, list):
            material_class_by_layer = np.asarray([int(v) for v in material_class_raw], dtype=np.int16)
            assert material_class_by_layer.size == nz, (
                "mock.material_class_by_layer length must equal mock.nz"
            )
        else:
            material_class_by_layer = np.full(nz, int(material_class_raw), dtype=np.int16)

        materials_raw = raw.get("materials", {})
        assert isinstance(materials_raw, dict), "materials must be a mapping"
        all_raw = materials_raw.get("all", {})
        assert isinstance(all_raw, dict), "materials.all must be a mapping"
        hk_default = float(all_raw.get("horizontal_conductivity", 1.0e-6))
        kv_default = float(all_raw.get("vertical_conductivity", hk_default))
        por_default = float(all_raw.get("porosity", 0.3))
        vga_default = float(all_raw.get("vG_alpha", 1.0))
        vgn_default = float(all_raw.get("vG_n", 2.0))

        hk_by_layer = _as_layer_array("hk_by_layer", hk_default)
        kv_by_layer = _as_layer_array("kv_by_layer", kv_default)
        porosity_by_layer = _as_layer_array("porosity_by_layer", por_default)
        vg_alpha_by_layer = _as_layer_array("vg_alpha_by_layer", vga_default)
        vg_n_by_layer = _as_layer_array("vg_n_by_layer", vgn_default)

        assert n_material_ids <= nz, "Too many material ids for provided layers"
        return MockConfig(
            config_path=config_path,
            workspace=workspace,
            model_name=model_name,
            grid_output_path=grid_output_path,
            material_output_path=material_output_path,
            sim_name=sim_name,
            nx=nx,
            ny=ny,
            nz=nz,
            delr=delr,
            delc=delc,
            delz=delz,
            top=top,
            layer_names=layer_names,
            material_by_layer=material_by_layer,
            material_class_by_layer=material_class_by_layer,
            hk_by_layer=hk_by_layer,
            kv_by_layer=kv_by_layer,
            porosity_by_layer=porosity_by_layer,
            vg_alpha_by_layer=vg_alpha_by_layer,
            vg_n_by_layer=vg_n_by_layer,
            material_name_by_layer=_as_material_names(),
            recharge_rate=float(all_raw.get("recharge_rate", 0.0)),
            specific_yield=float(all_raw.get("specific_yield", 0.1)),
            specific_storage=float(all_raw.get("specific_storage", 1.0e-5)),
        )


def build_mock_cube(config_path: Path) -> tuple[Path, Path]:
    cfg = MockConfig.from_yaml(config_path)
    cfg.grid_output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.material_output_path.parent.mkdir(parents=True, exist_ok=True)

    x_nodes = np.arange(cfg.nx + 1, dtype=float) * cfg.delr
    y_nodes = np.arange(cfg.ny + 1, dtype=float) * cfg.delc
    depth_cumsum = np.cumsum(cfg.delz)
    z_nodes = cfg.top - np.concatenate(([0.0], depth_cumsum))

    top = np.full((cfg.ny, cfg.nx), cfg.top, dtype=float)
    botm = top[None, :, :] - depth_cumsum[:, None, None]
    active_mask = np.ones((cfg.ny, cfg.nx), dtype=bool)
    idomain = np.ones((cfg.nz, cfg.ny, cfg.nx), dtype=int)

    materials = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=int)
    kh = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=float)
    kv = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=float)
    porosity = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=float)
    vg_alpha = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=float)
    vg_n = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=float)
    material_class = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=np.int16)

    for k in range(cfg.nz):
        materials[k] = int(cfg.material_by_layer[k])
        kh[k] = float(cfg.hk_by_layer[k])
        kv[k] = float(cfg.kv_by_layer[k])
        porosity[k] = float(cfg.porosity_by_layer[k])
        vg_alpha[k] = float(cfg.vg_alpha_by_layer[k])
        vg_n[k] = float(cfg.vg_n_by_layer[k])
        material_class[k] = np.int16(cfg.material_class_by_layer[k])

    step = np.asarray([cfg.delr, cfg.delc, float(cfg.delz[0])], dtype=float)
    el_dims = np.asarray([cfg.nx, cfg.ny, cfg.nz], dtype=int)
    origin = np.asarray([0.0, 0.0, float(z_nodes[-1])], dtype=float)

    np.savez_compressed(
        cfg.grid_output_path,
        origin=origin,
        step=step,
        z_thickness=cfg.delz,
        el_dims=el_dims,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        z_nodes=z_nodes,
        top=top,
        botm=botm,
        boundary_origin=np.asarray([0.0, 0.0], dtype=float),
        origin_global=np.asarray([0.0, 0.0, float(z_nodes[-1])], dtype=float),
        grid_corners_local=np.asarray([[0.0, 0.0], [x_nodes[-1], y_nodes[-1]]], dtype=float),
        grid_corners_global=np.asarray([[0.0, 0.0], [x_nodes[-1], y_nodes[-1]]], dtype=float),
        grid_corners_lonlat=np.asarray([[14.0, 50.0], [14.01, 50.01]], dtype=float),
        lonlat_epsg=4326,
        active_mask=active_mask,
        materials=materials,
        layer_names=np.asarray(cfg.layer_names, dtype=object),
    )

    np.savez_compressed(
        cfg.material_output_path,
        top=top,
        botm=botm,
        materials=materials,
        materials_mf=materials[::-1],
        active_mask=active_mask,
        idomain=idomain,
        kh=kh,
        kv=kv,
        hk=kh,
        k33=kv,
        porosity=porosity,
        van_genuchten_alpha=vg_alpha,
        van_genuchten_n=vg_n,
        material_class=material_class,
        material_class_by_layer=np.asarray(cfg.material_class_by_layer, dtype=np.int16),
        material_name_by_layer=np.asarray(cfg.material_name_by_layer, dtype=object),
        layer_names=np.asarray(cfg.layer_names, dtype=object),
        recharge_rate=np.asarray(cfg.recharge_rate, dtype=float),
        specific_yield=np.asarray(cfg.specific_yield, dtype=float),
        specific_storage=np.asarray(cfg.specific_storage, dtype=float),
    )

    LOG.info("Saved mock grid to %s", cfg.grid_output_path)
    LOG.info("Saved mock material parameters to %s", cfg.material_output_path)
    return cfg.grid_output_path, cfg.material_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple mock cube model inputs (no GIS).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/base/mock_cube.yaml"),
        help="Path to mock model config YAML",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    build_mock_cube(args.config)


if __name__ == "__main__":
    main()

