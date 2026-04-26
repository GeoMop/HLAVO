from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import hlavo.deep_model.model_3d_cfg as cfg
import hlavo.misc.config as cfg_common

LOG = logging.getLogger(__name__)


def _load_grid(npz_path: Path) -> dict[str, np.ndarray]:
    assert npz_path.exists(), f"Grid NPZ not found: {npz_path}"
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _vtk_cell_array(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 3, "Cell array must be 3D"
    cell_values = np.transpose(values, (2, 1, 0))
    return cell_values.ravel(order="F")


def export_grid_to_paraview(grid_npz: Path, output_path: Path) -> Path:
    import pyvista as pv

    data = _load_grid(grid_npz)
    x_nodes = data["x_nodes"].astype(float)
    y_nodes = data["y_nodes"].astype(float)
    z_nodes = data["z_nodes"].astype(float)
    materials = data["materials"].astype(int)
    active_mask = data["active_mask"].astype(bool)

    nx = x_nodes.size - 1
    ny = y_nodes.size - 1
    nz = z_nodes.size - 1

    assert materials.shape == (nz, ny, nx), "materials shape mismatch"
    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"

    grid = pv.RectilinearGrid(x_nodes, y_nodes, z_nodes)

    grid.cell_data["materials"] = _vtk_cell_array(materials)
    active_cells = np.broadcast_to(active_mask, (nz, ny, nx))
    grid.cell_data["active"] = _vtk_cell_array(active_cells.astype(np.uint8))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    assert output_path.exists(), f"Failed to write Paraview grid: {output_path}"
    LOG.info("Saved Paraview grid to %s", output_path)
    return output_path


def _resolve_workspace(config_path: Path) -> Path:
    raw, _ = cfg_common.load_config(config_path)
    common_raw = cfg.resolve_model_3d_common_raw(raw)
    workspace_root = cfg.resolve_workspace_root(None, common_raw)
    common = cfg.Model3DCommonConfig.from_mapping(common_raw)
    return cfg.resolve_model_workspace(workspace_root, common)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Modflow grid to Paraview VTK.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to the 3D YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override Paraview output path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    workspace = _resolve_workspace(args.config)
    grid_path = workspace / cfg.GRID_MATERIALS_FILENAME
    output_path = args.output if args.output is not None else workspace / "grid_materials.vtr"
    export_grid_to_paraview(grid_path, output_path)


if __name__ == "__main__":
    main()
