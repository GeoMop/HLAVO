from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import attrs
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_hlavo"))

import flopy
from . import model_3d_cfg as cfg3d
from .add_material_parameters import material_dataset_from_config, write_material_model_files
from .qgis_reader import BoundaryPolygon, GeometryConfig, Grid, ModelGeometry, ModelInputs, RasterLayer
from .simulation_builder import build_modflow_simulation

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class BuildConfig:
    workspace_root: Path
    common: cfg3d.Model3DCommonConfig
    geometry: GeometryConfig

    @classmethod
    def from_source(
        cls,
        config_source: Path | dict,
        workspace: Path | None = None,
    ) -> "BuildConfig":
        common, _, raw = cfg3d.Model3DCommonConfig.from_source(config_source)
        common_raw = cfg3d.resolve_model_3d_common_raw(raw)
        workspace_root = cfg3d.resolve_workspace_root(workspace, common_raw)
        geometry = GeometryConfig.from_source(config_source)
        return cls(
            workspace_root=workspace_root,
            common=common,
            geometry=geometry,
        )

    @property
    def output_path(self) -> Path:
        return self.geometry.resolve_grid_output_path(self.workspace)

    @property
    def resolved_material_parameters_path(self) -> Path:
        return self.workspace / cfg3d.MATERIAL_PARAMETERS_FILENAME

    @property
    def workspace(self) -> Path:
        return cfg3d.resolve_model_workspace(self.workspace_root, self.common)

    @property
    def model_name(self) -> str:
        return self.common.model_name

    @property
    def sim_name(self) -> str:
        return self.common.sim_name

    @property
    def exe_name(self) -> str:
        return self.common.exe_name

    @property
    def recharge_rate(self) -> float:
        return self.common.recharge_rate

    @property
    def drain_conductance(self) -> float:
        return self.common.drain_conductance

    # @property
    # def simulation_days(self) -> float:
    #     return self.common.simulation_days

    @property
    def stress_periods_days(self) -> tuple[float, ...]:
        return self.common.stress_periods_days


def build_model(
    config_source: Path | dict,
    workspace: Path | None = None,
) -> BuildConfig:
    build_config = BuildConfig.from_source(config_source, workspace=workspace)
    build_modflow_grid(config_source, build_config.output_path)
    write_material_model_files(config_source, workspace=workspace)
    write_modflow_inputs(build_config, config_source=config_source, workspace=workspace)
    return build_config


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


def materials_from_rasters_per_column(
    rasters: tuple[RasterLayer, ...],
    grid: Grid,
    active_mask: np.ndarray,
    top: np.ndarray,
) -> np.ndarray:
    ny = int(grid.el_dims[1])
    nx = int(grid.el_dims[0])
    nz = int(grid.el_dims[2])
    z_step = float(grid.step[2])

    z_centers = top[None, :, :] - z_step * (np.arange(nz, dtype=float)[:, None, None] + 0.5)
    materials = np.full((nz, ny, nx), -1, dtype=int)

    last_top = top - z_step * nz
    last_id = np.zeros((ny, nx), dtype=int)
    last_id[~active_mask] = -1

    rasters_bottom_up = tuple(reversed(rasters))
    for i_layer, raster in enumerate(rasters_bottom_up):
        layer_full = raster_to_full_grid(raster, grid)
        layer_data = np.ma.filled(layer_full, np.nan)
        layer_mask = np.ma.getmaskarray(layer_full)

        valid = active_mask & (~layer_mask) & (layer_data > last_top)
        if not np.any(valid):
            LOG.debug("Layer %s has no valid updates", raster.name)
            continue

        condition = (
            (z_centers >= last_top[None, :, :])
            & (z_centers < layer_data[None, :, :])
            & valid[None, :, :]
        )
        materials = np.where(condition, last_id[None, :, :], materials)

        last_top[valid] = layer_data[valid]
        last_id[valid] = i_layer
        LOG.debug(
            "Assigned layer %s (index %s) to %s cells (per-column)",
            raster.name,
            i_layer,
            int(condition.sum()),
        )

    return materials


def build_modflow_grid(config_source: Path | dict, output_path: Path) -> Path:
    geometry = ModelGeometry.from_source(config_source)
    geometry_path = geometry.write_npz(output_path)
    LOG.info("Saved geometry grid data to %s", geometry_path)
    return geometry_path


def _grid_arrays_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    assert npz_path.exists(), f"Grid NPZ not found: {npz_path}"
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _append_grid_lonlat_to_nam(nam_path: Path, grid_corners_lonlat: np.ndarray, epsg: int = 4326) -> None:
    assert grid_corners_lonlat.shape == (2, 2), "grid_corners_lonlat must be 2x2"
    sw = grid_corners_lonlat[0]
    ne = grid_corners_lonlat[1]
    marker = "Grid SW corner lon/lat"
    lines = [
        f"# {marker} (EPSG:{epsg}): {sw[1]:.6f}, {sw[0]:.6f}",
        f"# Grid NE corner lon/lat (EPSG:{epsg}): {ne[1]:.6f}, {ne[0]:.6f}",
    ]
    content = nam_path.read_text(encoding="utf-8")
    if marker in content:
        return
    existing = content.splitlines()
    insert_at = 1 if existing and existing[0].startswith("#") else 0
    updated = existing[:insert_at] + lines + existing[insert_at:]
    ending = "\n" if content.endswith("\n") else ""
    nam_path.write_text("\n".join(updated) + ending, encoding="utf-8")


def write_modflow_inputs(
    build_config: BuildConfig,
    *,
    config_source: Path | dict,
    workspace: Path | None = None,
) -> None:
    _ = _grid_arrays_from_npz(build_config.output_path)
    _ = _grid_arrays_from_npz(build_config.resolved_material_parameters_path)
    geometry = ModelGeometry.from_npz(build_config.output_path)
    material_dataset = material_dataset_from_config(config_source, workspace=workspace)
    build_modflow_simulation(
        common=build_config.common,
        geometry=geometry,
        material_dataset=material_dataset,
        workspace=build_config.workspace,
        exe_name=build_config.exe_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Modflow grid materials from QGIS inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Optional base workspace for generated model outputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    build_model(args.config, workspace=args.workspace)


if __name__ == "__main__":
    main()
