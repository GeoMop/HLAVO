from __future__ import annotations

import logging
from pathlib import Path
import shutil
import time
from threading import Thread
from ingress_server import app_config, worker
import pytest

workdir = Path(__file__).parent / "workdir"

def test_process_one_on_meteo_raw_data(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # AGENT: do not use tmp path for resulting data, use workdir instead to be able to review the
    # results after the test

    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "hlavo" / "ingress" / "scrapper" / "endpoints_config.yaml"
    data_dir = repo_root / "tests" / "ingress" / "scrapper" / "test_meteo_raw_data"

    # Ensure the test-only extractor module can be resolved.
    # JB: Want to test importing the extractor by ingress_server
    # no explicit Python path modification should be necessary
    #monkeypatch.syspath_prepend(str(Path(__file__).parent))

    # Use a local Zarr store to avoid S3 access.
    # Keep result in the workdir to be able to check the result
    monkeypatch.setenv("ZF_STORE_URL", str(workdir / "hlavo_test.zarr"))
    caplog.set_level(logging.INFO, logger="ingress_server")

    # copy source raw data to workdir simulatied raw data queue

    queue_dir = workdir / "queue"
    shutil.copytree(data_dir, queue_dir, dirs_exist_ok=True)
    app_cfg = app_config.load_app_config(config_path, queue_dir)
    # AGGENT: do not use app_cfg attributes to get subdirs since you want to test it
    # use e.g. `queue_dir / "accepted"` instead of `app_cfg.accepted_dir`

    input_files = [
        path
        for path in sorted((queue_dir / "accepted").rglob("*"))
        if path.is_file() and not path.name.endswith(".meta.json")
    ]
    assert input_files, "No input files found in accepted queue"

    thread = Thread(
        target=worker.working_loop,
        args=(app_cfg, 0.1),
        name="test-worker",
        daemon=True,
    )
    thread.start()

    remaining = list(input_files)
    print("# remaing files to process:", len(remaining))
    deadline = time.time() + 30
    while time.time() < deadline:
        remaining = [
            path
            for path in app_cfg.accepted_dir.rglob("*")
            if path.is_file() and not path.name.endswith(".meta.json")
        ]
        if not remaining:
            break
        time.sleep(0.1)

    app_cfg.stop_event.set()
    thread.join(timeout=5)

    worker_logs = caplog.text
    print(worker_logs)
