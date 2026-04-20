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
from .add_material_parameters import write_material_model_files
from .qgis_reader import BoundaryPolygon, GeometryConfig, Grid, ModelInputs, RasterLayer

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

    @property
    def simulation_days(self) -> float:
        return self.common.simulation_days

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
    write_modflow_inputs(build_config)
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
    model_inputs = ModelInputs.from_source(config_source)
    grid = model_inputs.grid
    boundary_origin = np.asarray(model_inputs.boundary.origin, dtype=float)

    grid_xy_min = grid.origin[:2].astype(float)
    grid_xy_max = grid_xy_min + grid.step[:2].astype(float) * grid.el_dims[:2].astype(float)
    grid_corners_local = np.asarray([grid_xy_min, grid_xy_max], dtype=float)
    grid_corners_global = grid_corners_local + boundary_origin[None, :]
    origin_global = np.asarray(
        [grid.origin[0] + boundary_origin[0], grid.origin[1] + boundary_origin[1], grid.origin[2]],
        dtype=float,
    )
    from pyproj import Transformer

    transformer = Transformer.from_crs(5514, 4326, always_xy=True)
    lon, lat = transformer.transform(grid_corners_global[:, 0], grid_corners_global[:, 1])
    grid_corners_lonlat = np.column_stack([lon, lat]).astype(float)

    active_mask = active_mask_from_boundary(model_inputs.boundary, grid)
    z_nodes = grid.z_nodes.astype(float)
    nz = int(grid.el_dims[2])
    assert z_nodes.size == nz + 1, "z_nodes size mismatch"
    top_raster = model_inputs.rasters[0]
    top_full = raster_to_full_grid(top_raster, grid)
    top = np.ma.filled(top_full, z_nodes[-1]).astype(float)
    z_step = float(grid.step[2])
    botm = top[None, :, :] - z_step * (np.arange(1, nz + 1, dtype=float)[:, None, None])

    materials = materials_from_rasters_per_column(
        model_inputs.rasters,
        grid,
        active_mask,
        top,
    )
    rasters_bottom_up = tuple(reversed(model_inputs.rasters))

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
        top=top,
        botm=botm,
        boundary_origin=boundary_origin,
        origin_global=origin_global,
        grid_corners_local=grid_corners_local,
        grid_corners_global=grid_corners_global,
        grid_corners_lonlat=grid_corners_lonlat,
        lonlat_epsg=4326,
        active_mask=active_mask,
        materials=materials,
        layer_names=np.asarray([r.name for r in rasters_bottom_up], dtype=object),
    )
    assert output_path.exists(), f"Grid output not created: {output_path}"
    LOG.info("Saved grid materials to %s", output_path)
    return output_path


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


def write_modflow_inputs(build_config: BuildConfig) -> None:
    # grid_data is the direct geometry export from build_modflow_grid():
    # grid extents, node coordinates, CRS metadata and the initial layer assignment.
    grid_data = _grid_arrays_from_npz(build_config.output_path)
    # model_data is the post-processed material model written by
    # write_material_model_files(). It keeps some geometry arrays duplicated on
    # purpose, but adds the MODFLOW-ready fields derived from materials
    # calibration/defaults such as idomain, hk and k33. When a field exists in
    # both files, prefer model_data here because it is the final source used for
    # the MODFLOW input packages.
    model_data = _grid_arrays_from_npz(build_config.resolved_material_parameters_path)
    el_dims = np.asarray(grid_data["el_dims"], dtype=int)
    assert el_dims.size == 3, "el_dims must contain 3 values"
    nx = int(el_dims[0])
    ny = int(el_dims[1])
    nz = int(el_dims[2])
    step = np.asarray(grid_data["step"], dtype=float)
    assert step.size == 3, "step must contain 3 values"

    active_mask = np.asarray(model_data["active_mask"], dtype=bool)
    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"
    top = np.asarray(model_data["top"], dtype=float)
    botm = np.asarray(model_data["botm"], dtype=float)
    materials = np.asarray(model_data["materials"], dtype=int)
    assert top.shape == (ny, nx), "top shape mismatch"
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"
    assert materials.shape == (nz, ny, nx), "materials shape mismatch"

    idomain = np.asarray(model_data["idomain"], dtype=int)
    hk = np.asarray(model_data["kh"] if "kh" in model_data else model_data["hk"], dtype=float)
    k33 = np.asarray(model_data["kv"] if "kv" in model_data else model_data["k33"], dtype=float)
    assert idomain.shape == (nz, ny, nx), "idomain shape mismatch"
    assert hk.shape == (nz, ny, nx), "hk shape mismatch"
    assert k33.shape == (nz, ny, nx), "k33 shape mismatch"

    workdir = build_config.workspace
    workdir.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(
        sim_name=build_config.sim_name,
        exe_name=build_config.exe_name,
        sim_ws=str(workdir),
    )
    perioddata = [(float(days), 1, 1.0) for days in build_config.stress_periods_days]
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(perioddata),
        perioddata=perioddata,
    )
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=build_config.sim_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nz,
        nrow=ny,
        ncol=nx,
        delr=float(step[0]),
        delc=float(step[1]),
        top=top,
        botm=botm,
        idomain=idomain,
    )

    layer_tops = np.empty_like(botm)
    layer_tops[0] = top
    if nz > 1:
        layer_tops[1:] = botm[:-1]
    flopy.mf6.ModflowGwfic(gwf, strt=layer_tops)

    icelltype = np.zeros(nz, dtype=int)
    icelltype[0] = 1
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=hk, k33=k33, save_specific_discharge=True)

    recharge_rate = (
        float(model_data["recharge_rate"]) if "recharge_rate" in model_data else build_config.recharge_rate
    )
    recharge = np.where(active_mask, recharge_rate, 0.0)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)

    top_active = active_mask & (idomain[0] > 0)
    top_rows, top_cols = np.where(top_active)
    drain_cells = []
    for row, col in zip(top_rows.tolist(), top_cols.tolist()):
        drain_cells.append((0, int(row), int(col), float(top[row, col]), build_config.drain_conductance))
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drain_cells)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{build_config.sim_name}.hds",
        budget_filerecord=f"{build_config.sim_name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    sim.write_simulation()

    grid_corners_lonlat = grid_data.get("grid_corners_lonlat")
    if grid_corners_lonlat is not None:
        lonlat_epsg = int(grid_data.get("lonlat_epsg", 4326))
        _append_grid_lonlat_to_nam(workdir / "mfsim.nam", np.asarray(grid_corners_lonlat, dtype=float), lonlat_epsg)
        _append_grid_lonlat_to_nam(
            workdir / f"{build_config.sim_name}.nam",
            np.asarray(grid_corners_lonlat, dtype=float),
            lonlat_epsg,
        )
    LOG.info("Saved MODFLOW input files to %s", workdir)


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
