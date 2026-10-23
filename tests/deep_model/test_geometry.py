"""Deep model geometry (hlavo/deep_model/geometry.py).

The synthetic tests need no GIS data (they run on CI); `test_from_gis` reads the QGIS project of
runs/composed_3D_only and is skipped when the GIS rasters are not available (DVC).
"""
from pathlib import Path

import numpy as np
import pytest

from hlavo.deep_model.geometry import GEOMETRY_FILE, NO_MATERIAL, VTK_FILE, Geometry

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "runs/composed_3D_only/config.yaml"
GIS_RASTER = REPO_ROOT / "hlavo/deep_model/GIS/geology_model_used/DEM_Turow_10m_resample.tif"

JTSK_ORIGIN = np.array([-700000.0, -961000.0])
XY_MIN = np.array([-200.0, -2920.0])


def _synthetic() -> Geometry:
    """3 x 4 columns, dz = 10 m. Units bottom-up: "A" (base 0 m), "B" (base 20 m, missing in
    column (1, 1)), "relief" (40 m, 35 m in column (0, 0)); column (2, 3) outside the boundary."""
    nrow, ncol = 3, 4
    base_a = np.zeros((nrow, ncol))
    base_b = np.full((nrow, ncol), 20.0)
    base_b[1, 1] = np.nan
    relief = np.full((nrow, ncol), 40.0)
    relief[0, 0] = 35.0
    active = np.ones((nrow, ncol), dtype=bool)
    active[2, 3] = False
    x0, y0 = XY_MIN
    ring = np.array([[x0, y0], [x0 + 140, y0], [x0 + 140, y0 + 120], [x0, y0 + 120], [x0, y0]])
    return Geometry.from_surfaces(
        jtsk_origin=JTSK_ORIGIN,
        xy_min=XY_MIN,
        step=np.array([40.0, 40.0, 10.0]),
        boundary=ring,
        active_columns=active,
        surfaces=np.stack([base_a, base_b, relief]),
        material_names=("A", "B", "relief"),
    )


def test_material_assignment():
    geometry = _synthetic()
    assert geometry.shape == (4, 3, 4)  # elevations 0..40 m in layers of 10 m
    assert geometry.material_names == ("A", "B", "relief")
    # Regular column: B from 40 m down to its base 20 m, A below.
    np.testing.assert_array_equal(geometry.material[:, 1, 2], [1, 1, 0, 0])
    np.testing.assert_array_equal(geometry.botm[:, 1, 2], [30.0, 20.0, 10.0, 0.0])
    # B missing: A up to the relief.
    np.testing.assert_array_equal(geometry.material[:, 1, 1], [0, 0, 0, 0])
    # Lower relief: layers shifted down with the top (terrain-following).
    np.testing.assert_array_equal(geometry.botm[:, 0, 0], [25.0, 15.0, 5.0, -5.0])
    np.testing.assert_array_equal(geometry.material[:, 0, 0], [1, 1, 0, 0])
    # Column outside the boundary.
    np.testing.assert_array_equal(geometry.material[:, 2, 3], [NO_MATERIAL] * 4)
    assert geometry.active.sum() == 4 * (3 * 4 - 1)


def test_save_load_roundtrip(tmp_path):
    geometry = _synthetic()
    path = geometry.save(tmp_path)
    assert path == tmp_path / GEOMETRY_FILE and path.exists()

    loaded = Geometry.load(tmp_path)
    for name in ("jtsk_origin", "xy_min", "step", "top", "material", "surfaces", "boundary"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(geometry, name), err_msg=name)
    assert loaded.material_names == geometry.material_names


def test_write_vtk(tmp_path):
    import pyvista as pv

    geometry = _synthetic()
    path = geometry.write_vtk(tmp_path)
    assert path == tmp_path / VTK_FILE

    cells = pv.read(path)["cells"]
    assert cells.n_cells == 4 * 3 * 4
    material = np.asarray(cells.cell_data["material"])
    assert np.sum(material == NO_MATERIAL) == 4
    assert np.sum(material == 1) == np.sum(geometry.material == 1)
    # Right-handed cells (index order x, y, z all increasing): positive volumes.
    assert np.all(cells.compute_cell_sizes()["Volume"] > 0.0)


def test_coordinates():
    geometry = _synthetic()
    np.testing.assert_array_equal(geometry.local_to_jtsk([0.0, 0.0]), JTSK_ORIGIN)
    np.testing.assert_array_equal(geometry.jtsk_to_local(JTSK_ORIGIN + [10.0, 20.0]), [10.0, 20.0])

    # SW grid corner of the Uhelna model; lon/lat as stored by the former build_model
    # (runs/composed_3D_only/model_with_mine/grid_materials.npz, pyproj EPSG:5514 -> 4326).
    sw_lonlat = geometry.local_to_lonlat(XY_MIN)
    np.testing.assert_allclose(sw_lonlat, [14.86749592, 50.84325147], atol=1e-7)

    # Round trip local -> lon/lat -> local for a few points (shape preserved). The S-JTSK <-> WGS84
    # datum shift of pyproj is not exactly invertible: about 1.3 mm error, irrelevant for the grid.
    points = np.array([[0.0, 0.0], [1500.0, -800.0], [5000.0, 2000.0]])
    np.testing.assert_allclose(geometry.lonlat_to_local(geometry.local_to_lonlat(points)), points, atol=0.01)

    # The Uhelna 1D site lies inside the model grid extent.
    x, y = geometry.lonlat_to_local([14.889853, 50.863565])
    assert XY_MIN[0] < x < XY_MIN[0] + 137 * 40.0 and XY_MIN[1] < y < XY_MIN[1] + 124 * 40.0


@pytest.mark.skipif(not GIS_RASTER.exists(), reason=f"GIS rasters not available: {GIS_RASTER}")
def test_from_gis(tmp_path):
    geometry = Geometry.from_gis(RUN_CONFIG)
    nlay, nrow, ncol = geometry.shape
    assert geometry.top.shape == (nrow, ncol)
    assert geometry.active.any()
    assert geometry.material.max() < len(geometry.material_names)

    geometry.save(tmp_path)
    loaded = Geometry.load(tmp_path)
    np.testing.assert_array_equal(loaded.material, geometry.material)
    assert loaded.material_names == geometry.material_names
