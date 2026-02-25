from __future__ import annotations

import argparse
import logging
from pathlib import Path

import attrs
import numpy as np
import yaml

from qgis_reader import BoundaryPolygon, Grid, ModelInputs, RasterLayer

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class BuildConfig:
    config_path: Path
    output_path: Path

    @staticmethod
    def from_yaml(config_path: Path, output_path: Path | None = None) -> "BuildConfig":
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict), "Config YAML must be a mapping"
        model_raw = raw.get("model", {})
        assert isinstance(model_raw, dict), "model config must be a mapping"
        model_name = model_raw.get("model_name")
        assert model_name, "model.model_name is required in the config"
        model_name = str(model_name)
        base_workspace = Path(model_raw.get("workspace", "model"))
        run_workspace = base_workspace / model_name

        output_raw = raw.get("grid_output_path")
        if output_path is None:
            if output_raw:
                output_path = Path(str(output_raw))
                if not output_path.is_absolute():
                    output_path = run_workspace / output_path
            else:
                output_path = run_workspace / "grid_materials.npz"
        return BuildConfig(config_path=config_path, output_path=output_path)


def active_mask_from_rasters(
    rasters: tuple[RasterLayer, ...], grid: Grid
) -> np.ndarray:
    assert rasters, "Rasters are required to build active mask"
    mask = None
    reference_mask = None
    for raster in rasters:
        full = raster_to_full_grid(raster, grid)
        raster_mask = np.ma.getmaskarray(full)
        if reference_mask is None:
            reference_mask = raster_mask
        else:
            if not np.array_equal(raster_mask, reference_mask):
                diff = np.logical_xor(raster_mask, reference_mask)
                diff_count = int(diff.sum())
                LOG.warning(
                    "Raster mask mismatch for %s: %s cells differ",
                    raster.name,
                    diff_count,
                )
        if mask is None:
            mask = raster_mask.copy()
        else:
            mask = mask & raster_mask
    assert mask is not None, "Active mask could not be derived from rasters"
    return ~mask


def active_mask_from_boundary(boundary: BoundaryPolygon, grid: Grid) -> np.ndarray:
    from shapely.geometry import Polygon, mapping
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    ny = int(grid.el_dims[1])
    nx = int(grid.el_dims[0])
    step_x = float(grid.step[0])
    step_y = float(grid.step[1])
    assert step_x > 0.0 and step_y > 0.0, "Grid steps must be positive"

    x_min = float(grid.origin[0])
    y_max = float(grid.origin[1]) + step_y * ny
    transform = from_origin(x_min, y_max, step_x, step_y)

    polygon = Polygon(boundary.coords_local)
    assert polygon.is_valid, "Boundary polygon is invalid"
    inside = geometry_mask(
        [mapping(polygon)],
        out_shape=(ny, nx),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return inside


def raster_to_full_grid(raster: RasterLayer, grid: Grid) -> np.ma.MaskedArray:
    ny = int(grid.el_dims[1])
    nx = int(grid.el_dims[0])
    full = np.ma.masked_all((ny, nx), dtype=float)

    step_x = float(raster.pixel_size[0])
    step_y = float(raster.pixel_size[1])
    assert step_x != 0.0 and step_y != 0.0, "Raster pixel size cannot be zero"

    x_min, y_min = raster.extent_local[0]
    x_max, y_max = raster.extent_local[1]
    x_start = x_min if step_x > 0 else x_max
    y_start = y_max if step_y < 0 else y_min

    height, width = raster.z_field.shape
    x_coords = x_start + (np.arange(width, dtype=float) + 0.5) * step_x
    y_coords = y_start + (np.arange(height, dtype=float) + 0.5) * step_y

    grid_step_x = float(grid.step[0])
    grid_step_y = float(grid.step[1])
    grid_x0 = float(grid.origin[0]) + 0.5 * grid_step_x
    grid_y0 = float(grid.origin[1]) + grid_step_y * ny - 0.5 * grid_step_y

    ix = np.round((x_coords - grid_x0) / grid_step_x).astype(int)
    iy = np.round((grid_y0 - y_coords) / grid_step_y).astype(int)

    assert ix.size == width, "Raster width mismatch"
    assert iy.size == height, "Raster height mismatch"
    assert ix.min() >= 0 and ix.max() < nx, "Raster X extent outside grid"
    assert iy.min() >= 0 and iy.max() < ny, "Raster Y extent outside grid"

    full[np.ix_(iy, ix)] = raster.z_field
    return full


def assign_materials(
    rasters_bottom_up: tuple[RasterLayer, ...],
    grid: Grid,
    active_mask: np.ndarray,
) -> np.ndarray:
    ny = int(grid.el_dims[1])
    nx = int(grid.el_dims[0])
    nz = int(grid.el_dims[2])
    z_step = float(grid.step[2])

    z_centers = grid.z_nodes[:-1] + 0.5 * z_step
    materials = np.full((nz, ny, nx), -1, dtype=int)

    last_top = np.full((ny, nx), float(grid.origin[2]), dtype=float)
    last_id = np.zeros((ny, nx), dtype=int)
    last_id[~active_mask] = -1

    for i_layer, raster in enumerate(rasters_bottom_up):
        layer_full = raster_to_full_grid(raster, grid)
        layer_data = np.ma.filled(layer_full, np.nan)
        layer_mask = np.ma.getmaskarray(layer_full)

        valid = active_mask & (~layer_mask) & (layer_data > last_top)
        if not np.any(valid):
            LOG.debug("Layer %s has no valid updates", raster.name)
            continue

        start_z = np.round(last_top / z_step) * z_step
        condition = (
            (z_centers[:, None, None] >= start_z[None, :, :])
            & (z_centers[:, None, None] < layer_data[None, :, :])
            & valid[None, :, :]
        )
        materials = np.where(condition, last_id[None, :, :], materials)

        last_top[valid] = layer_data[valid]
        last_id[valid] = i_layer
        LOG.debug(
            "Assigned layer %s (index %s) to %s cells",
            raster.name,
            i_layer,
            int(condition.sum()),
        )

    return materials


def build_modflow_grid(config_path: Path, output_path: Path) -> Path:
    model_inputs = ModelInputs.from_yaml(config_path)
    grid = model_inputs.grid

    active_mask = active_mask_from_boundary(model_inputs.boundary, grid)
    full_active = np.ones_like(active_mask, dtype=bool)
    rasters_bottom_up = tuple(reversed(model_inputs.rasters))

    materials_full = assign_materials(rasters_bottom_up, grid, full_active)
    materials = np.where(active_mask[None, :, :], materials_full, -1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        origin=grid.origin,
        step=grid.step,
        el_dims=grid.el_dims,
        x_nodes=grid.x_nodes,
        y_nodes=grid.y_nodes,
        z_nodes=grid.z_nodes,
        active_mask=active_mask,
        materials=materials,
        layer_names=np.asarray([r.name for r in rasters_bottom_up], dtype=object),
    )
    assert output_path.exists(), f"Grid output not created: {output_path}"
    LOG.info("Saved grid materials to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Modflow grid materials from QGIS inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the grid npz",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    build_config = BuildConfig.from_yaml(args.config, args.output)
    build_modflow_grid(build_config.config_path, build_config.output_path)


if __name__ == "__main__":
    main()
