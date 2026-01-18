from __future__ import annotations

import logging
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from functools import cached_property

import attrs
import numpy as np
import yaml

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class RasterLayer:
    """Raster data masked to the boundary polygon."""

    name: str
    # Layer name from the QGIS project.
    source: str
    # Resolved raster source path.
    crs_wkt: str
    # CRS WKT string, if available.
    extent_local: "np.ndarray"
    # Local extent as [[xmin, ymin], [xmax, ymax]].
    pixel_size: "np.ndarray"
    # Pixel size (x, y) in map units.
    size: "np.ndarray"
    # Raster dimensions (width, height).
    z_field: "np.ma.MaskedArray"
    # Raster values masked to the boundary polygon with nodata masked out.

    @cached_property
    def z_extent(self) -> "np.ndarray":
        data = self.z_field
        return np.asarray([data.min(), data.max()])


@attrs.define(frozen=True)
class BoundaryPolygon:
    """Boundary polygon ring in source CRS coordinates."""

    raw_ring: "np.ndarray"
    # Polygon exterior ring as Nx2 array of XY coordinates.

    @property
    def origin(self) -> tuple[float, float]:
        first_point = self.raw_ring[0]
        return (_round_to_km(first_point[0]), _round_to_km(first_point[1]))

    @cached_property
    def coords_local(self) -> "np.ndarray":
        return self.raw_ring - self.origin


@attrs.define(frozen=True)
class ModelInputs:
    """In-memory representation of boundary and raster inputs."""

    boundary: BoundaryPolygon
    # Model boundary polygon.
    rasters: tuple[RasterLayer, ...]
    # Raster layers in project order.

    @staticmethod
    def from_yaml(config_path: Path) -> "ModelInputs":
        config = ModelConfig.from_yaml(config_path)
        reader = QgisProjectReader(
            project_path=config.qgis_project_path,
            boundary_layer_name=config.boundary_layer_name,
            raster_group_name=config.raster_group_name,
        )
        return reader.read()


@attrs.define(frozen=True)
class ModelConfig:
    """Configuration loaded from the model YAML file."""

    qgis_project_path: Path
    # Path to the QGIS project (.qgs/.qgz).
    boundary_layer_name: str
    # Layer name of the boundary polygon.
    raster_group_name: str
    # Layer tree group containing model rasters.
    meshsteps: tuple[float, float, float]
    # Mesh steps (x, y, z) in model units.

    @staticmethod
    def from_yaml(path: Path) -> "ModelConfig":
        assert path.exists(), f"Config file not found: {path}"
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        assert isinstance(raw, dict), "Config YAML must be a mapping"
        assert "qgis_project_path" in raw, "Missing required config key: qgis_project_path"
        qgis_project_path = Path(raw["qgis_project_path"])
        boundary_layer_name = str(raw.get("boundary_layer_name", "JB_extended_domain"))
        raster_group_name = str(raw.get("raster_group_name", "HG model layers"))
        assert "meshsteps" in raw, "Missing required config key: meshsteps"
        meshsteps_raw = raw["meshsteps"]
        assert isinstance(meshsteps_raw, dict), "meshsteps must be a mapping with x, y, z"
        meshsteps = (
            float(meshsteps_raw["x"]),
            float(meshsteps_raw["y"]),
            float(meshsteps_raw["z"]),
        )

        return ModelConfig(
            qgis_project_path=qgis_project_path,
            boundary_layer_name=boundary_layer_name,
            raster_group_name=raster_group_name,
            meshsteps=meshsteps,
        )


@attrs.define(frozen=True)
class QgisProjectReader:
    """Read boundary and raster metadata directly from a QGIS project file."""

    project_path: Path
    # Path to the QGIS project.
    boundary_layer_name: str = "JB_extended_domain"
    # Layer name of the boundary polygon.
    raster_group_name: str = "HG model layers"
    # Layer tree group containing model rasters.

    def read(self) -> ModelInputs:
        project_path = self.project_path
        assert project_path.exists(), f"Missing QGIS project: {project_path}"

        root, project_dir, home_path = _load_project_xml(project_path)
        boundary = self._read_boundary_xml(root, project_dir, home_path)
        rasters = self._read_rasters_xml(root, project_dir, home_path, boundary)
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
        boundary: BoundaryPolygon,
    ) -> tuple[RasterLayer, ...]:
        import rasterio

        tree_root = root.find("layer-tree-group")
        assert tree_root is not None, "Missing layer-tree-group in QGIS project"
        group = _find_layer_tree_group(tree_root, self.raster_group_name)
        assert group is not None, f"Raster group not found: {self.raster_group_name}"

        maplayers = _maplayers_by_id(root)
        raster_layers: list[RasterLayer] = []
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
                _assert_krovak_crs(dataset.crs)
                bounds = dataset.bounds
                extent = np.array(
                    [[bounds.left, bounds.bottom], [bounds.right, bounds.top]]
                )
                assert dataset.nodata is not None, "Raster nodata value is missing"
                data = dataset.read(1).astype(float)
                if dataset.nodata != -1000:
                    data[data == dataset.nodata] = -1000.0
                z_field = _mask_raster_with_boundary(data, dataset.transform, boundary)
                crs_wkt = dataset.crs.to_wkt() if dataset.crs else ""
                extent_local = extent - boundary.origin
                raster_layers.append(
                    RasterLayer(
                        name=layer_name or _require_text(maplayer, "layername"),
                        source=str(resolved_source),
                        crs_wkt=crs_wkt,
                        extent_local=extent_local,
                        pixel_size=np.asarray([dataset.res[0], dataset.res[1]]),
                        size=np.asarray([dataset.width, dataset.height]),
                        z_field=z_field,
                    )
                )

        assert raster_layers, f"No raster layers found in group: {self.raster_group_name}"
        return tuple(raster_layers)


def _round_to_km(value: float) -> float:
    return round(value / 1000.0) * 1000.0




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


def _mask_raster_with_boundary(
    data: "np.ndarray", transform: "object", boundary: BoundaryPolygon
) -> "np.ma.MaskedArray":
    from shapely.geometry import Polygon, mapping
    from rasterio.features import geometry_mask

    polygon = Polygon(boundary.raw_ring)
    assert polygon.is_valid, "Boundary polygon is invalid"
    inside = geometry_mask(
        [mapping(polygon)],
        invert=True,
        out_shape=data.shape,
        transform=transform,
        all_touched=True,
    )
    mask = ~inside | (data == -1000.0)
    return np.ma.array(data, mask=mask)


def _assert_krovak_crs(crs: "object") -> None:
    assert crs is not None, "Raster CRS is missing"
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if epsg == 5514:
        return
    crs_text = ""
    if hasattr(crs, "to_wkt"):
        crs_text = crs.to_wkt()
    elif hasattr(crs, "to_string"):
        crs_text = crs.to_string()
    assert "EPSG\",\"5514" in crs_text or "EPSG:5514" in crs_text, (
        f"Expected Krovak EPSG:5514, got {epsg or crs_text}"
    )
