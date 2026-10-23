"""Geometry of the deep (3D) model: structured grid, terrain-following layers, geologic units.

`Geometry` is created once from the GIS project (`Geometry.from_gis`, on the desktop: needs the
QGIS project and its rasters), saved into a folder (`save`, one file `geometry.nc`) and recreated
from that folder alone (`load`), e.g. on a cluster. Inspect it in ParaView via `write_vtk`.
It holds no physical material properties and no boundary conditions; those are added when the
MODFLOW simulation is built (milestone 5).

Coordinate systems
- local:  model XY [m] = S-JTSK coordinates minus `jtsk_origin` (the first point of the boundary
          polygon rounded to km); the grid and the VTK output use it.
- S-JTSK: Krovak East North (EPSG:5514), the CRS of the GIS rasters and of the boundary polygon.
- lon/lat: WGS84 (EPSG:4326) degrees.

Grid: shape (nlay, nrow, ncol) in MODFLOW order; row 0 is the northernmost row, col 0 the
westernmost column. The layers follow the terrain: layer 0 starts at the relief `top` of its
column and every layer has the thickness dz, so botm[k] = top - (k + 1) * dz. The number of
layers covers the elevation range of all raster surfaces.

Geologic units (materials): the HG rasters are the base surfaces of the units. Ordered bottom-up,
unit i fills a column from its base surface i up to the base surface of the next unit that lies
higher in that column; the lowest unit also fills the column down to the model bottom. Cells
outside the boundary polygon have the material index -1.
"""
from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import xarray as xr

from hlavo.deep_model.qgis_reader import BoundaryPolygon, Grid, ModelInputs, RasterLayer

LOG = logging.getLogger(__name__)

JTSK_EPSG = 5514
LONLAT_EPSG = 4326
GEOMETRY_FILE = "geometry.nc"
VTK_FILE = "geometry.vtm"
NO_MATERIAL = -1


@attrs.define(frozen=True, eq=False)
class Geometry:
    """Grid, layer elevations and geologic unit of every cell (see the module doc).

    jtsk_origin:    (2,) S-JTSK coordinates [m] of the local origin (0, 0).
    xy_min:         (2,) local XY [m] of the south-west corner of the grid.
    step:           (3,) cell size dx, dy, dz [m].
    top:            (nrow, ncol) elevation [m a.s.l.] of the model top (relief) of every column.
    material:       (nlay, nrow, ncol) int32, index into `material_names`; `NO_MATERIAL` (-1)
                    for cells outside the model.
    material_names: names of the geologic units = names of the HG raster layers, bottom-up.
    surfaces:       (n_units, nrow, ncol) elevation [m a.s.l.] of the base surface of every unit
                    at the column centres, NaN where the raster has no data.
    boundary:       (n_points, 2) local XY [m] of the boundary polygon ring.
    """

    jtsk_origin: np.ndarray
    xy_min: np.ndarray
    step: np.ndarray
    top: np.ndarray
    material: np.ndarray
    material_names: tuple[str, ...]
    surfaces: np.ndarray
    boundary: np.ndarray

    def __attrs_post_init__(self) -> None:
        nlay, nrow, ncol = self.material.shape
        assert self.top.shape == (nrow, ncol), f"top shape {self.top.shape} != {(nrow, ncol)}"
        assert self.surfaces.shape == (len(self.material_names), nrow, ncol), (
            f"surfaces shape {self.surfaces.shape} != {(len(self.material_names), nrow, ncol)}"
        )
        assert self.jtsk_origin.shape == (2,) and self.xy_min.shape == (2,) and self.step.shape == (3,)
        assert self.material.max() < len(self.material_names), "material index out of material_names"

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def from_gis(cls, config_source: Path | dict) -> "Geometry":
        """Read the QGIS project given by the `model_3d.geometry` config section.

        config_source: path of the YAML config (or its loaded mapping); see `GeometryConfig` in
        qgis_reader.py for the keys. The rasters are resampled to the grid columns there.
        """
        inputs = ModelInputs.from_source(config_source)
        grid = inputs.grid
        rasters_bottom_up = tuple(reversed(inputs.rasters))
        geometry = cls.from_surfaces(
            jtsk_origin=np.asarray(inputs.boundary.origin, dtype=float),
            xy_min=np.asarray(grid.origin[:2], dtype=float),
            step=np.asarray(grid.step, dtype=float),
            boundary=inputs.boundary.coords_local,
            active_columns=active_mask_from_boundary(inputs.boundary, grid),
            surfaces=np.stack([np.ma.filled(raster_to_full_grid(r, grid), np.nan) for r in rasters_bottom_up]),
            material_names=tuple(r.name for r in rasters_bottom_up),
        )
        assert geometry.material.shape[0] == int(grid.el_dims[2]), "Layer count differs from the GIS grid"
        LOG.info("Geometry from %s: shape %s, %d units", config_source, geometry.shape, len(geometry.material_names))
        return geometry

    @classmethod
    def from_surfaces(
        cls,
        jtsk_origin: np.ndarray,
        xy_min: np.ndarray,
        step: np.ndarray,
        boundary: np.ndarray,
        active_columns: np.ndarray,
        surfaces: np.ndarray,
        material_names: tuple[str, ...],
    ) -> "Geometry":
        """Build the geometry from unit base surfaces given on the grid columns (no GIS needed).

        active_columns: (nrow, ncol) bool, columns inside the boundary polygon.
        surfaces:       (n_units, nrow, ncol) base surfaces bottom-up, NaN = no data; the last one
                        is the relief, it gives `top` (NaN filled by the grid top elevation).
        Other arguments as the attributes of `Geometry`.
        """
        dz = float(step[2])
        z_min = np.floor(np.nanmin(surfaces) / dz) * dz
        z_max = np.ceil(np.nanmax(surfaces) / dz) * dz
        nlay = int(round((z_max - z_min) / dz))
        assert nlay > 0, "Surfaces must span at least one layer thickness dz"
        top = np.where(np.isfinite(surfaces[-1]), surfaces[-1], z_max)
        return cls(
            jtsk_origin=np.asarray(jtsk_origin, dtype=float),
            xy_min=np.asarray(xy_min, dtype=float),
            step=np.asarray(step, dtype=float),
            top=top,
            material=_assign_material(top, dz, nlay, np.asarray(active_columns, dtype=bool), surfaces),
            material_names=tuple(str(name) for name in material_names),
            surfaces=np.asarray(surfaces, dtype=float),
            boundary=np.asarray(boundary, dtype=float),
        )

    # ---- grid ------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        """(nlay, nrow, ncol)"""
        return self.material.shape

    @cached_property
    def botm(self) -> np.ndarray:
        """(nlay, nrow, ncol) bottom elevation [m a.s.l.] of every cell (terrain-following layers)."""
        nlay = self.shape[0]
        return self.top[None, :, :] - self.step[2] * np.arange(1, nlay + 1)[:, None, None]

    @property
    def active(self) -> np.ndarray:
        """(nlay, nrow, ncol) bool, cells inside the model (material assigned)."""
        return self.material != NO_MATERIAL

    @cached_property
    def x_centers(self) -> np.ndarray:
        """(ncol,) local x [m] of the column centres, west to east."""
        return self.xy_min[0] + (np.arange(self.shape[2]) + 0.5) * self.step[0]

    @cached_property
    def y_centers(self) -> np.ndarray:
        """(nrow,) local y [m] of the row centres, north to south (row 0 is the northernmost)."""
        nrow = self.shape[1]
        return self.xy_min[1] + (nrow - np.arange(nrow) - 0.5) * self.step[1]

    # ---- coordinates -----------------------------------------------------------------------
    def local_to_jtsk(self, xy: np.ndarray) -> np.ndarray:
        """Local XY [m], shape (..., 2) -> S-JTSK (EPSG:5514) [m], same shape."""
        return np.asarray(xy, dtype=float) + self.jtsk_origin

    def jtsk_to_local(self, xy: np.ndarray) -> np.ndarray:
        """S-JTSK (EPSG:5514) [m], shape (..., 2) -> local XY [m], same shape."""
        return np.asarray(xy, dtype=float) - self.jtsk_origin

    def local_to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        """Local XY [m], shape (..., 2) -> (longitude, latitude) [deg] WGS84, same shape."""
        return _transform(self.local_to_jtsk(xy), JTSK_EPSG, LONLAT_EPSG)

    def lonlat_to_local(self, lonlat: np.ndarray) -> np.ndarray:
        """(longitude, latitude) [deg] WGS84, shape (..., 2) -> local XY [m], same shape.

        Inverse of `local_to_lonlat` up to about 1 mm (the S-JTSK <-> WGS84 datum shift of pyproj
        is not exactly invertible).
        """
        return self.jtsk_to_local(_transform(np.asarray(lonlat, dtype=float), LONLAT_EPSG, JTSK_EPSG))

    # ---- persistence -----------------------------------------------------------------------
    def to_dataset(self) -> xr.Dataset:
        """All attributes as an xarray Dataset (the content of `geometry.nc`)."""
        return xr.Dataset(
            data_vars={
                "top": (("row", "col"), self.top, {"description": "model top (relief) [m a.s.l.]"}),
                "material": (("layer", "row", "col"), self.material.astype(np.int32),
                             {"description": "index of the geologic unit, -1 outside the model"}),
                "surface": (("unit", "row", "col"), self.surfaces,
                            {"description": "base surface of the unit [m a.s.l.], NaN = no data"}),
                "boundary": (("boundary_point", "xy"), self.boundary,
                             {"description": "boundary polygon ring, local XY [m]"}),
            },
            coords={
                "unit": ("unit", list(self.material_names), {"description": "geologic units, bottom-up"}),
                "x": ("col", self.x_centers, {"description": "local x of the column centre [m]"}),
                "y": ("row", self.y_centers, {"description": "local y of the row centre [m]"}),
            },
            attrs={
                "description": "HLAVO deep model geometry, see hlavo/deep_model/geometry.py",
                "jtsk_epsg": JTSK_EPSG,
                "jtsk_origin_x": float(self.jtsk_origin[0]),
                "jtsk_origin_y": float(self.jtsk_origin[1]),
                "x_min": float(self.xy_min[0]),
                "y_min": float(self.xy_min[1]),
                "dx": float(self.step[0]),
                "dy": float(self.step[1]),
                "dz": float(self.step[2]),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "Geometry":
        """Inverse of `to_dataset`."""
        attrs_ = ds.attrs
        return cls(
            jtsk_origin=np.array([attrs_["jtsk_origin_x"], attrs_["jtsk_origin_y"]], dtype=float),
            xy_min=np.array([attrs_["x_min"], attrs_["y_min"]], dtype=float),
            step=np.array([attrs_["dx"], attrs_["dy"], attrs_["dz"]], dtype=float),
            top=ds["top"].values.astype(float),
            material=ds["material"].values.astype(np.int32),
            material_names=tuple(str(name) for name in ds["unit"].values),
            surfaces=ds["surface"].values.astype(float),
            boundary=ds["boundary"].values.astype(float),
        )

    def save(self, folder: Path) -> Path:
        """Write `geometry.nc` (netCDF 3, scipy engine) into `folder`; returns its path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / GEOMETRY_FILE
        self.to_dataset().to_netcdf(path, engine="scipy")
        LOG.info("Saved geometry to %s", path)
        return path

    @classmethod
    def load(cls, folder: Path) -> "Geometry":
        """Recreate the geometry from `folder/geometry.nc` written by `save` (no GIS needed)."""
        path = Path(folder) / GEOMETRY_FILE
        assert path.exists(), f"Geometry file not found: {path}"
        return cls.from_dataset(xr.load_dataset(path, engine="scipy"))

    def write_vtk(self, folder: Path) -> Path:
        """Write `geometry.vtm` (VTK multiblock, local XY) into `folder` for ParaView; returns its path.

        Blocks: `cells` - the model grid with the cell data `material` (and `active`); node
        elevations are the mean top of the adjacent columns (display only, the cells themselves
        have one top per column); `surface_<unit>` - base surface of every unit at the column
        centres (NaN where missing); `boundary` - the boundary polygon at the highest model top.
        """
        import pyvista as pv

        nlay, nrow, ncol = self.shape
        dx, dy, dz = self.step
        # VTK index order (i, j, k) = (x, y, z), all increasing (right-handed cells): rows flipped
        # to south -> north, layers flipped to bottom -> top.
        padded = np.pad(self.top[::-1, :], 1, mode="edge")
        top_nodes = 0.25 * (padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:])
        x_nodes = self.xy_min[0] + np.arange(ncol + 1) * dx
        y_nodes = self.xy_min[1] + np.arange(nrow + 1) * dy
        z_nodes = top_nodes[None, :, :] - dz * np.arange(nlay, -1, -1)[:, None, None]  # (nlay+1, nrow+1, ncol+1)
        x, y = np.meshgrid(x_nodes, y_nodes)  # (nrow+1, ncol+1)
        cells = pv.StructuredGrid(
            *(np.broadcast_to(a, z_nodes.shape).transpose(2, 1, 0) for a in (x, y, z_nodes))
        )
        material_ijk = self.material[::-1, ::-1, :].transpose(2, 1, 0)
        cells.cell_data["material"] = material_ijk.ravel(order="F")
        cells.cell_data["active"] = (material_ijk != NO_MATERIAL).ravel(order="F").astype(np.uint8)

        blocks = pv.MultiBlock()
        blocks["cells"] = cells
        xc, yc = np.meshgrid(self.x_centers, self.y_centers)
        for name, surface in zip(self.material_names, self.surfaces):
            blocks[f"surface_{name}"] = pv.StructuredGrid(xc, yc, surface)
        ring = np.column_stack([self.boundary, np.full(len(self.boundary), np.nanmax(self.top))])
        blocks["boundary"] = pv.lines_from_points(ring)

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / VTK_FILE
        blocks.save(str(path))
        LOG.info("Saved geometry VTK to %s", path)
        return path


def _assign_material(top: np.ndarray, dz: float, nlay: int, active_columns: np.ndarray,
                     surfaces: np.ndarray) -> np.ndarray:
    """Geologic unit index of every cell (nlay, nrow, ncol), see the module doc.

    Bottom-up over the unit base surfaces: the cells of a column between the last valid surface
    below and the current surface get the unit of the last surface (the lowest unit also the
    cells below its base). A surface is valid in a column if it has data there and lies above
    the last valid surface.
    """
    z_center = top[None, :, :] - dz * (np.arange(nlay)[:, None, None] + 0.5)
    material = np.full(z_center.shape, NO_MATERIAL, dtype=np.int32)
    last_top = top - dz * nlay
    last_id = np.where(active_columns, 0, NO_MATERIAL)
    for i_unit, surface in enumerate(surfaces):
        valid = active_columns & (surface > last_top)  # NaN (no data) compares False
        in_unit = (z_center >= last_top) & (z_center < surface) & valid
        material = np.where(in_unit, last_id, material)
        last_top = np.where(valid, surface, last_top)
        last_id = np.where(valid, i_unit, last_id)
    return material.astype(np.int32)


def _transform(xy: np.ndarray, from_epsg: int, to_epsg: int) -> np.ndarray:
    """Transform points (..., 2) between EPSG coordinate systems, (x, y) = (easting, northing) or (lon, lat)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x, y = transformer.transform(xy[..., 0], xy[..., 1])
    return np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=-1)


# ---- GIS rasters and boundary on the grid (moved from build_modflow_grid.py) ------------------
def active_mask_from_boundary(boundary: BoundaryPolygon, grid: Grid) -> np.ndarray:
    """(nrow, ncol) bool, grid columns touched by the boundary polygon (row 0 = north)."""
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
    return geometry_mask(
        [mapping(polygon)],
        out_shape=(ny, nx),
        transform=transform,
        invert=True,
        all_touched=True,
    )


def raster_to_full_grid(raster: RasterLayer, grid: Grid) -> np.ma.MaskedArray:
    """Raster resampled to the grid columns (cropped by qgis_reader) placed on the full (nrow, ncol) grid."""
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
