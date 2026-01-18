from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET
from functools import cached_property

import attrs
import numpy as np
import yaml

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
    raw_ring: "np.ndarray"

    @property
    def origin(self) -> tuple[float, float]:
        first_point = self.raw_ring[0]
        return (_round_to_km(first_point[0]), _round_to_km(first_point[1]))

    @cached_property
    def coords_local(self) -> "np.ndarray":
        return self.raw_ring - self.origin


@attrs.define(frozen=True)
class ModelInputs:
    boundary: BoundaryPolygon
    rasters: tuple[RasterLayerInfo, ...]

    @staticmethod
    def from_yaml(config_path: Path) -> "ModelInputs":
        config = ModelConfig.from_yaml(config_path)
        reader = QgisProjectReader(
            project_path=config.qgis_project_path,
            boundary_layer_name=config.boundary_layer_name,
            raster_group_name=config.raster_group_name,
            qgis_prefix=config.qgis_prefix,
        )
        return reader.read()


@attrs.define(frozen=True)
class ModelConfig:
    qgis_project_path: Path
    boundary_layer_name: str
    raster_group_name: str
    qgis_prefix: Path | None
    meshsteps: tuple[float, float, float]

    @staticmethod
    def from_yaml(path: Path) -> "ModelConfig":
        assert path.exists(), f"Config file not found: {path}"
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        assert isinstance(raw, dict), "Config YAML must be a mapping"
        qgis_project_path = Path(_require_key(raw, "qgis_project_path"))
        boundary_layer_name = str(raw.get("boundary_layer_name", "JB_extended_domain"))
        raster_group_name = str(raw.get("raster_group_name", "HG model layers"))
        qgis_prefix_value = raw.get("qgis_prefix")
        qgis_prefix = Path(qgis_prefix_value) if qgis_prefix_value else None

        meshsteps_raw = _require_key(raw, "meshsteps")
        assert isinstance(meshsteps_raw, dict), "meshsteps must be a mapping with x, y, z"
        meshsteps = (
            float(_require_key(meshsteps_raw, "x")),
            float(_require_key(meshsteps_raw, "y")),
            float(_require_key(meshsteps_raw, "z")),
        )

        return ModelConfig(
            qgis_project_path=qgis_project_path,
            boundary_layer_name=boundary_layer_name,
            raster_group_name=raster_group_name,
            qgis_prefix=qgis_prefix,
            meshsteps=meshsteps,
        )


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

    def read(self) -> ModelInputs:
        project_path = self.project_path
        assert project_path.exists(), f"Missing QGIS project: {project_path}"

        root, project_dir, home_path = _load_project_xml(project_path)
        boundary = self._read_boundary_xml(root, project_dir, home_path)
        rasters = self._read_rasters_xml(root, project_dir, home_path, boundary.origin)
        LOG.debug("Loaded boundary with single ring")
        LOG.debug("Loaded %s raster layers from group %s", len(rasters), self.raster_group_name)
        LOG.debug("Local origin set to %s", boundary.origin)

        return ModelInputs(boundary=boundary, rasters=rasters)

    def _read_boundary_xml(
        self, root: ET.Element, project_dir: Path, home_path: Path | None
    ) -> BoundaryPolygon:
        import geopandas as gpd

        maplayers = _maplayers_by_id(root)
        layer = _find_maplayer_by_name(maplayers, self.boundary_layer_name)
        assert layer is not None, f"Boundary layer not found: {self.boundary_layer_name}"

        datasource = _require_text(layer, "datasource")
        source_path_str, layer_name = _split_datasource(datasource)
        resolved_source = _resolve_source_path(source_path_str, project_dir, home_path)
        LOG.debug("Boundary source resolved to %s", resolved_source)

        if layer_name:
            gdf = gpd.read_file(resolved_source, layer=layer_name)
        else:
            gdf = gpd.read_file(resolved_source)

        assert len(gdf) > 0, f"Boundary layer {self.boundary_layer_name} has no features"
        assert len(gdf) == 1, f"Expected one boundary feature, found {len(gdf)}"
        geom = gdf.geometry.iloc[0]
        assert geom is not None and not geom.is_empty, "Boundary geometry is empty"

        if geom.geom_type == "MultiPolygon":
            geoms = list(geom.geoms)
            assert geoms, "Boundary multipolygon has no parts"
            if len(geoms) > 1:
                LOG.debug("Boundary multipolygon has %s parts, selecting largest", len(geoms))
            geom = max(geoms, key=lambda g: g.area)

        assert geom.geom_type == "Polygon", f"Boundary geometry must be polygon, got {geom.geom_type}"
        interiors = list(geom.interiors)
        assert not interiors, f"Boundary polygon has {len(interiors)} interior rings"
        raw_ring = np.asarray([(float(x), float(y)) for x, y in geom.exterior.coords])
        return BoundaryPolygon(raw_ring=raw_ring)

    def _read_rasters_xml(
        self,
        root: ET.Element,
        project_dir: Path,
        home_path: Path | None,
        origin: tuple[float, float],
    ) -> tuple[RasterLayerInfo, ...]:
        import rasterio

        tree_root = root.find("layer-tree-group")
        assert tree_root is not None, "Missing layer-tree-group in QGIS project"
        group = _find_layer_tree_group(tree_root, self.raster_group_name)
        assert group is not None, f"Raster group not found: {self.raster_group_name}"

        maplayers = _maplayers_by_id(root)
        raster_layers: list[RasterLayerInfo] = []
        for layer_id, layer_name in _iter_layer_tree_layers(group):
            maplayer = maplayers.get(layer_id)
            assert maplayer is not None, f"Raster maplayer not found for id {layer_id}"
            if maplayer.attrib.get("type") != "raster":
                continue

            datasource = _require_text(maplayer, "datasource")
            source_path_str, _ = _split_datasource(datasource)
            resolved_source = _resolve_source_path(source_path_str, project_dir, home_path)
            assert resolved_source, f"Raster source path missing for layer {layer_name}"

            with rasterio.open(resolved_source) as dataset:
                bounds = dataset.bounds
                extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
                local_extent = _extent_to_local(extent, origin)
                crs_wkt = dataset.crs.to_wkt() if dataset.crs else ""
                raster_layers.append(
                    RasterLayerInfo(
                        name=layer_name or _require_text(maplayer, "layername"),
                        source=str(resolved_source),
                        crs_wkt=crs_wkt,
                        extent=extent,
                        local_extent=local_extent,
                        pixel_size=(float(dataset.res[0]), float(dataset.res[1])),
                        size=(int(dataset.width), int(dataset.height)),
                    )
                )

        assert raster_layers, f"No raster layers found in group: {self.raster_group_name}"
        return tuple(raster_layers)


def describe_inputs(data: ModelInputs) -> str:
    lines = [
        f"Local origin: {data.boundary.origin}",
        "Boundary rings: 1",
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


def _round_to_km(value: float) -> float:
    return round(value / 1000.0) * 1000.0


def _extent_to_local(extent: tuple[float, float, float, float], origin: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0 = origin
    return (
        extent[0] - x0,
        extent[1] - x0,
        extent[2] - y0,
        extent[3] - y0,
    )




def _require_key(data: dict, key: str) -> object:
    if key not in data:
        raise AssertionError(f"Missing required config key: {key}")
    return data[key]


def _load_project_xml(project_path: Path) -> tuple[ET.Element, Path, Path | None]:
    if project_path.suffix.lower() == ".qgz":
        with zipfile.ZipFile(project_path) as archive:
            name = _find_project_xml_name(archive.namelist())
            xml_bytes = archive.read(name)
    else:
        xml_bytes = project_path.read_bytes()

    root = ET.fromstring(xml_bytes)
    project_dir = project_path.parent
    home_path = _project_home_path(root, project_dir)
    return root, project_dir, home_path


def _find_project_xml_name(names: list[str]) -> str:
    for name in names:
        if name.lower().endswith(".qgs"):
            return name
    assert names, "Empty QGIS project archive"
    return names[0]


def _project_home_path(root: ET.Element, project_dir: Path) -> Path | None:
    home = root.find("homePath")
    if home is None:
        return None
    raw = home.attrib.get("path")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def _maplayers_by_id(root: ET.Element) -> dict[str, ET.Element]:
    layers: dict[str, ET.Element] = {}
    for layer in root.findall(".//maplayer"):
        layer_id = layer.findtext("id")
        if layer_id:
            layers[layer_id] = layer
    return layers


def _find_maplayer_by_name(maplayers: dict[str, ET.Element], name: str) -> ET.Element | None:
    for layer in maplayers.values():
        layer_name = layer.findtext("layername") or layer.findtext("name")
        if layer_name == name:
            return layer
    return None


def _require_text(element: ET.Element, tag: str) -> str:
    value = element.findtext(tag)
    assert value is not None and value != "", f"Missing {tag} in QGIS project for layer"
    return value


def _find_layer_tree_group(root: ET.Element, name: str) -> ET.Element | None:
    if root.tag == "layer-tree-group" and root.attrib.get("name") == name:
        return root
    for child in root:
        if child.tag == "layer-tree-group":
            found = _find_layer_tree_group(child, name)
            if found is not None:
                return found
    return None


def _iter_layer_tree_layers(root: ET.Element) -> list[tuple[str, str]]:
    layers: list[tuple[str, str]] = []

    def walk(node: ET.Element) -> None:
        for child in node:
            if child.tag == "layer-tree-layer":
                layer_id = child.attrib.get("id")
                name = child.attrib.get("name")
                if layer_id:
                    layers.append((layer_id, name or ""))
            elif child.tag == "layer-tree-group":
                walk(child)

    walk(root)
    return layers


def _split_datasource(datasource: str) -> tuple[str, str | None]:
    data = datasource
    if data.startswith("file:"):
        data = data[len("file:") :]
    parts = data.split("|")
    path_part = parts[0]
    layer_name = None
    for part in parts[1:]:
        if part.startswith("layername="):
            layer_name = part[len("layername=") :]
    if "?" in path_part:
        path_part = path_part.split("?", 1)[0]
    return path_part, layer_name


def _resolve_source_path(path_str: str, project_dir: Path, home_path: Path | None) -> str:
    if path_str.startswith("/vsizip/"):
        inner = path_str[len("/vsizip/") :]
        if inner.startswith("./") or inner.startswith("../"):
            resolved = (project_dir / inner).resolve()
            return "/vsizip/" + str(resolved)
        return path_str

    candidate = Path(path_str)
    if candidate.is_absolute():
        assert candidate.exists(), f"Missing data source: {candidate}"
        return str(candidate)

    project_candidate = (project_dir / candidate).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    if home_path is not None:
        home_candidate = (home_path / candidate).resolve()
        if home_candidate.exists():
            return str(home_candidate)

    assert project_candidate.exists(), f"Missing data source: {project_candidate}"
    return str(project_candidate)
