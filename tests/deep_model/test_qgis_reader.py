from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from qgis_reader import ModelInputs, write_vtk_surfaces

def _config_path() -> Path:
    config = SCRIPT_DIR.parent / "model_config.yaml"
    assert config.exists(), f"Config file not found: {config}"
    return config

def test_qgis_project_reader():
    data = ModelInputs.from_yaml(_config_path())

    assert data.boundary.raw_ring.size > 0
    assert isinstance(data.boundary.raw_ring, np.ndarray)
    ring = data.boundary.raw_ring
    assert ring.ndim == 2 and ring.shape[1] == 2, "Ring must be Nx2 array"
    mean_xy = ring.mean(axis=0)
    print(f"ring 1: n_points={ring.shape[0]} mean=({mean_xy[0]:.3f}, {mean_xy[1]:.3f})")
    assert len(data.rasters) > 0
    for idx, raster in enumerate(data.rasters, start=1):
        extent = raster.extent_local
        x_min, y_min = extent[0]
        x_max, y_max = extent[1]
        print(
            "raster %s: %s z_extent=%s shape=%s extent_local=((%.3f, %.3f), (%.3f, %.3f))"
            % (
                idx,
                raster.name,
                raster.z_extent,
                raster.z_field.shape,
                x_min,
                y_min,
                x_max,
                y_max,
            )
        )
        z_field = raster.z_field
        assert np.ma.isMaskedArray(z_field)
        corners = [
            z_field[0, 0],
            z_field[0, -1],
            z_field[-1, 0],
            z_field[-1, -1],
        ]
        assert any(value is np.ma.masked for value in corners), (
            f"Raster {raster.name} missing -1000 nodata in corners"
        )
    grid = data.grid
    assert grid.x_nodes.size > 1
    assert grid.y_nodes.size > 1
    assert grid.z_nodes.size > 1
    assert np.allclose(np.diff(grid.x_nodes), np.diff(grid.x_nodes)[0])
    assert np.allclose(np.diff(grid.y_nodes), np.diff(grid.y_nodes)[0])
    assert np.allclose(np.diff(grid.z_nodes), np.diff(grid.z_nodes)[0])
    grid_extent = (
        (float(grid.x_nodes.min()), float(grid.y_nodes.min())),
        (float(grid.x_nodes.max()), float(grid.y_nodes.max())),
    )
    print(
        "grid extent_local=((%.3f, %.3f), (%.3f, %.3f)) shape=(%s, %s, %s)"
        % (
            grid_extent[0][0],
            grid_extent[0][1],
            grid_extent[1][0],
            grid_extent[1][1],
            grid.x_nodes.size,
            grid.y_nodes.size,
            grid.z_nodes.size,
        )
    )


def test_vtk_surface_export() -> None:
    data = ModelInputs.from_yaml(_config_path())
    model_dir = ROOT.parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for child in model_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.iterdir():
                if nested.is_file():
                    nested.unlink()
    vtk_path = model_dir / "surfaces.vtm"
    result = write_vtk_surfaces(data, vtk_path)
    assert result.exists()
    assert result.stat().st_size > 0
    import pyvista as pv

    blocks = pv.read(str(result))
    for idx, raster in enumerate(data.rasters, start=1):
        block_name = f"layer_{idx}_{raster.name}"
        surface = blocks[block_name]
        assert surface is not None, f"Missing VTK block: {block_name}"
        bounds = surface.bounds
        x_min, y_min = raster.extent_local[0]
        x_max, y_max = raster.extent_local[1]
        step_x, step_y = raster.pixel_size
        step_x = abs(float(step_x))
        step_y = abs(float(step_y))
        x_low, x_high = (x_min, x_max) if x_min <= x_max else (x_max, x_min)
        y_low, y_high = (y_min, y_max) if y_min <= y_max else (y_max, y_min)
        expected_x_min = x_low + 0.5 * step_x
        expected_x_max = x_high - 0.5 * step_x
        expected_y_min = y_low + 0.5 * step_y
        expected_y_max = y_high - 0.5 * step_y
        assert np.isclose(bounds[0], expected_x_min)
        assert np.isclose(bounds[1], expected_x_max)
        assert np.isclose(bounds[2], expected_y_min)
        assert np.isclose(bounds[3], expected_y_max)
