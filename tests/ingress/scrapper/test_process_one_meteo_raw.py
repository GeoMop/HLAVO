from __future__ import annotations

import importlib
from pathlib import Path
import ingress_server as igs
import pytest

workdir = Path(__file__).parent / "workdir"

def test_process_one_on_meteo_raw_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "hlavo" / "ingress" / "scrapper" / "endpoints_config.yaml"
    data_dir = repo_root / "tests" / "ingress" / "scrapper" / "test_meteo_raw_data"

    # Ensure the test-only extractor module can be resolved.
    #monkeypatch.syspath_prepend(str(Path(__file__).parent))

    # Use a local Zarr store to avoid S3 access.
    monkeypatch.setenv("ZF_STORE_URL", str(workdir / "hlavo_test.zarr"))

    igs.app_config.load_app_config(config_path, data_dir)
    app_config = igs.prepare_test_environment()
    igs.worker.working_loop(app_config)