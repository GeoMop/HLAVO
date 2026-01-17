from __future__ import annotations

import os
from pathlib import Path

import pytest

from qgis_reader import QgisProjectReader


def _project_path() -> Path:
    gis_dir = os.environ.get("GIS_DIR")
    if not gis_dir:
        raise AssertionError(
            "GIS_DIR env var not set. Set GIS_DIR to the read-only GIS directory."
        )

    project = Path(gis_dir) / "uhelna_all.qgz"
    assert project.exists(), f"QGIS project not found: {project}"
    return project


def test_qgis_project_reader():
    project = _project_path()
    reader = QgisProjectReader(project_path=project)
    data = reader.read()

    assert data.boundary.name == "JB_extended_domain"
    assert data.boundary.polygon_local.rings
    assert len(data.rasters) > 0
