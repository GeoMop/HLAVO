from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from hlavo.composed import model_composed

TESTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TESTS_DIR / "test_composed_config.yaml"


def test_setup_models_uses_real_dask_queues_with_constant_weather_model(tmp_path: Path, dask_client) -> None:
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    config_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime_config_path = tmp_path / "test_composed_runtime_config.yaml"
    runtime_config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    final_state = model_composed.setup_models(
        work_dir=work_dir,
        config_path=runtime_config_path,
        client=dask_client,
    )

    assert final_state == np.datetime64("2025-03-06T02:00:00")
