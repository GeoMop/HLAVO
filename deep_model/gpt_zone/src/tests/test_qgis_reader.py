from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from qgis_reader import ModelInputs

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
        print(f"raster {idx}: {raster.name}")
