"""Milestone 3: MODFLOW 6 through modflowapi inside the composed 1D-3D loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from hlavo.composed import model_composed
from hlavo.composed.model_3d_api import default_libmf6

N_DAYS = 10
RECHARGE = {0: 0.01, 1: 0.0}  # m/day from the constant 1D models


def _write_config(tmp_path: Path) -> Path:
    config = {
        "seed": 123456,
        "start_datetime": "2025-03-01T00:00:00",
        "end_datetime": f"2025-03-{1 + N_DAYS:02d}T00:00:00",
        "model_3d": {
            "backend_class_name": "Model3DAPI",
            "common": {
                "name": "uhelna",
                "time_step_hours": 24.0,
                "ims": {"complexity": "MODERATE"},
                "model_builder": "modflow_cube:build_cube",
                "model_builder_config": {"initial_head": -60.0},
                "writer": {"class_name": "FilePredictionWriter"},
            },
        },
        "model_1d": {
            "model_1d_class_name": "Model1DConstantWeather",
            "site_ids": [0, 1],
            "sites": [
                {"longitude": 14.88, "latitude": 50.86, "velocity": RECHARGE[0]},
                {"longitude": 14.90, "latitude": 50.88, "velocity": RECHARGE[1]},
            ],
        },
    }
    config_path = tmp_path / "model_3d_api_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_model_3d_api_couples_modflow_with_1d_models(tmp_path, dask_client):
    assert default_libmf6().exists()
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    config_path = _write_config(tmp_path)

    final_time = model_composed.setup_models(work_dir=work_dir, config_path=config_path, client=dask_client)

    assert final_time == np.datetime64(f"2025-03-{1 + N_DAYS:02d}T00:00:00")
    assert (work_dir / "model_3d" / "mfsim.nam").exists()

    rows = [json.loads(line) for line in (work_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    site_rows = [row for row in rows if row["node"] == "site_prediction"]
    assert len(site_rows) == N_DAYS * 2

    heads = {site: np.array([r["pressure_head"] for r in site_rows if r["site_id"] == site]) for site in (0, 1)}
    for site, values in heads.items():
        assert values.shape == (N_DAYS,)
        assert np.all(np.isfinite(values)), f"non-finite heads for site {site}: {values}"
        assert np.allclose([r["velocity"] for r in site_rows if r["site_id"] == site], RECHARGE[site])

    # Recharged west half rises steadily above the initial -60 m; the unrecharged
    # east half (next to the -60 m constant head) stays close to it and below site 0.
    assert np.all(np.diff(heads[0]) > 0), heads[0]
    assert heads[0][-1] > -60.0 + 0.1, heads[0]
    assert heads[0][-1] < -60.0 + N_DAYS * RECHARGE[0] / 0.1, heads[0]  # no more than storage allows
    assert np.all(heads[1] >= -60.0 - 1e-6), heads[1]
    assert np.all(heads[1] < heads[0]), (heads[0], heads[1])
