from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import sys

import attrs
import numpy as np

LOG = logging.getLogger(__name__)


def _ensure_pyqgis(qgis_prefix: Path | None) -> None:
    if importlib.util.find_spec("qgis") is not None:
        return

    if qgis_prefix is None:
        qgis_prefix = Path(os.environ.get("QGIS_PREFIX_PATH", "/usr"))

    candidate_paths = [
        qgis_prefix / "share" / "qgis" / "python",
        Path("/usr/lib/python3/dist-packages"),
    ]

    os.environ.setdefault("QGIS_PREFIX_PATH", str(qgis_prefix))
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))

    if importlib.util.find_spec("qgis") is None:
        raise RuntimeError(
            "PyQGIS not found. Set QGIS_PREFIX_PATH or PYTHONPATH to include QGIS Python modules."
        )


@attrs.define(frozen=True)
class RasterLayerInfo:
    name: str
    source: str
    crs_wkt: str
    extent: tuple[float, float, float, float]
    local_extent: tuple[float, float, float, float]
    pixel_size: tuple[float, float]
    size: tuple[int, int]


@attrs.define(frozen=True)
class BoundaryPolygon:
    rings: tuple["np.ndarray", ...]


@attrs.define(frozen=True)
class BoundaryPolygonInfo:
    name: str
    crs_wkt: str
    origin: tuple[float, float]
    polygon_local: BoundaryPolygon


@attrs.define(frozen=True)
class QgisProjectData:
    project_path: Path
    crs_wkt: str
    local_origin: tuple[float, float]
    boundary: BoundaryPolygonInfo
    rasters: tuple[RasterLayerInfo, ...]


@attrs.define(frozen=True)
class QgisSession:
    qgis_prefix: Path | None = None
    _app: object | None = attrs.field(init=False, default=None, repr=False)

    def __enter__(self) -> "QgisSession":
        _ensure_pyqgis(self.qgis_prefix)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qgis.core import QgsApplication

        prefix = str(self.qgis_prefix or Path(os.environ.get("QGIS_PREFIX_PATH", "/usr")))
        QgsApplication.setPrefixPath(prefix, True)
        app = QgsApplication([], False)
        app.initQgis()
        object.__setattr__(self, "_app", app)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        from qgis.core import QgsApplication

        if hasattr(self, "_app"):
            QgsApplication.exitQgis()
        return False


@attrs.define(frozen=True)
class QgisProjectReader:
    project_path: Path
    boundary_layer_name: str = "JB_extended_domain"
    raster_group_name: str = "HG model layers"
    qgis_prefix: Path | None = None

    def read(self) -> QgisProjectData:
        project_path = self.project_path
        assert project_path.exists(), f"Missing QGIS project: {project_path}"

        with QgisSession(qgis_prefix=self.qgis_prefix):
            from qgis.core import QgsProject

            project = QgsProject()
            ok = project.read(str(project_path))
            assert ok, f"Failed to read QGIS project: {project_path}"

            boundary = self._read_boundary(project)
            rasters = self._read_rasters(project, boundary.origin)
            LOG.debug("Loaded boundary %s", boundary.name)
            LOG.debug("Loaded %s raster layers from group %s", len(rasters), self.raster_group_name)
            LOG.debug("Local origin set to %s", boundary.origin)

            crs = project.crs()
            crs_wkt = crs.toWkt() if crs.isValid() else ""
            return QgisProjectData(
                project_path=project_path,
                crs_wkt=crs_wkt,
                local_origin=boundary.origin,
                boundary=boundary,
                rasters=rasters,
            )

    def _read_boundary(self, project: "QgsProject") -> BoundaryPolygonInfo:
        from qgis.core import QgsWkbTypes

        layers = project.mapLayersByName(self.boundary_layer_name)
        assert layers, f"Boundary layer not found: {self.boundary_layer_name}"
        assert len(layers) == 1, f"Expected one boundary layer, found {len(layers)}"

        layer = layers[0]
        assert layer.isValid(), "Boundary layer is invalid"
        assert layer.geometryType() == QgsWkbTypes.PolygonGeometry, "Boundary is not a polygon layer"

        geometries = []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            assert geom is not None and not geom.isEmpty(), "Boundary geometry is empty"
            geometries.append(geom)

        assert len(geometries) == 1, f"Expected one boundary feature, found {len(geometries)}"
        geometry = geometries[0]
        crs = layer.crs()
        crs_wkt = crs.toWkt() if crs.isValid() else ""
        origin = _local_origin_from_geometry(geometry)
        polygon_local = _geometry_to_local_polygon(geometry, origin)

        return BoundaryPolygonInfo(
            name=layer.name(),
            crs_wkt=crs_wkt,
            origin=origin,
            polygon_local=polygon_local,
        )

    def _read_rasters(
        self, project: "QgsProject", origin: tuple[float, float]
    ) -> tuple[RasterLayerInfo, ...]:
        from qgis.core import QgsLayerTreeLayer, QgsMapLayer, QgsRasterLayer

        root = project.layerTreeRoot()
        group = root.findGroup(self.raster_group_name)
        assert group is not None, f"Raster group not found: {self.raster_group_name}"

        raster_layers: list[RasterLayerInfo] = []
        for child in group.children():
            if not isinstance(child, QgsLayerTreeLayer):
                continue
            layer = child.layer()
            if layer is None or layer.type() != QgsMapLayer.RasterLayer:
                continue

            assert isinstance(layer, QgsRasterLayer)
            assert layer.isValid(), f"Raster layer invalid: {layer.name()}"

            extent = layer.extent()
            local_extent = _extent_to_local(extent, origin)
            raster_layers.append(
                RasterLayerInfo(
                    name=layer.name(),
                    source=layer.source(),
                    crs_wkt=layer.crs().toWkt() if layer.crs().isValid() else "",
                    extent=(extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()),
                    local_extent=local_extent,
                    pixel_size=(layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY()),
                    size=(layer.width(), layer.height()),
                )
            )

        assert raster_layers, f"No raster layers found in group: {self.raster_group_name}"
        return tuple(raster_layers)


def describe_project(data: QgisProjectData) -> str:
    lines = [
        f"Project: {data.project_path}",
        f"CRS: {data.crs_wkt[:120]}",
        f"Local origin: {data.local_origin}",
        f"Boundary: {data.boundary.name}",
        f"Rasters: {len(data.rasters)}",
    ]
    for raster in data.rasters:
        lines.append(
            (
                f"- {raster.name} size={raster.size} pixel={raster.pixel_size} "
                f"extent={raster.extent} local_extent={raster.local_extent}"
            )
        )
    return "\n".join(lines)


def _local_origin_from_geometry(geom: "QgsGeometry") -> tuple[float, float]:
    polygon = _polygon_from_geometry(geom)
    assert polygon, "Boundary geometry has no polygon rings"
    first_ring = polygon[0]
    assert first_ring, "Boundary polygon has empty ring"
    first_point = first_ring[0]
    x = float(first_point.x())
    y = float(first_point.y())
    return (_round_to_km(x), _round_to_km(y))


def _round_to_km(value: float) -> float:
    return round(value / 1000.0) * 1000.0


def _extent_to_local(extent: "QgsRectangle", origin: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0 = origin
    return (
        extent.xMinimum() - x0,
        extent.xMaximum() - x0,
        extent.yMinimum() - y0,
        extent.yMaximum() - y0,
    )


def _geometry_to_local_polygon(geom: "QgsGeometry", origin: tuple[float, float]) -> BoundaryPolygon:
    origin_arr = np.array(origin, dtype=float)
    rings = _polygon_from_geometry(geom)
    assert rings, "Boundary geometry has no polygon rings"
    return _rings_to_local_polygon(rings, origin_arr)


def _polygon_from_geometry(geom: "QgsGeometry") -> list[list["QgsPointXY"]]:
    if geom.isMultipart():
        raise AssertionError("Boundary geometry is multipart; expected single polygon")

    polygon = geom.asPolygon()
    assert polygon, "Boundary geometry polygon is empty"
    return polygon


def _rings_to_local_polygon(
    rings: list[list["QgsPointXY"]], origin: "np.ndarray"
) -> BoundaryPolygon:
    local_rings: list["np.ndarray"] = []
    for ring in rings:
        assert ring, "Boundary polygon ring is empty"
        coords = np.array([(point.x(), point.y()) for point in ring], dtype=float)
        local_rings.append(coords - origin)
    return BoundaryPolygon(rings=tuple(local_rings))
