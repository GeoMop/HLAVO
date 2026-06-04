#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path

from hlavo.composed.model_composed import run_simulation
from hlavo.deep_model.build_modflow_grid import build_model
from hlavo.misc.logging_utils import ensure_debug_file_handler, ensure_stdout_handler, set_hlavo_loggers
from hlavo.tools import zf

LOG = logging.getLogger(__name__)


def _resolve_workdir(config_path: Path, workdir: str | Path | None) -> Path:
    if workdir is None:
        return config_path.parent / "workdir"
    return Path(workdir).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HLAVO command line interface.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    build_parser = subparsers.add_parser(
        "build_model",
        help="Build the 3D MODFLOW model from GIS resources.",
    )
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run the coupled 1D-3D simulation.",
    )
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Print overview of all dataset nodes defined by project schemas.",
    )

    for subparser in (build_parser, simulate_parser):
        subparser.add_argument("config_file", type=Path, help="Path to the YAML config file.")
        subparser.add_argument(
            "-w",
            "--workdir",
            type=Path,
            default=None,
            help='Optional working directory. Defaults to "<config_dir>/workdir".',
        )

    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def configure_calculation_logging(workdir: Path, log_name: str = "hlavo_main.log", reset: bool = False) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / log_name
    if reset:
        log_path.unlink(missing_ok=True)

    set_hlavo_loggers()
    resolved_log_path = ensure_debug_file_handler(log_path, "_hlavo_main_log_handler")
    ensure_stdout_handler("_hlavo_stdout_handler")
    return resolved_log_path


def _run_build_model(*, config_path: Path, workdir: Path) -> int:
    build_config = build_model(config_source=config_path, workspace=workdir)
    LOG.info("3D model built in %s", build_config.workspace)
    return 0


def _run_simulate(*, config_path: Path, workdir: Path) -> int:
    log_path = configure_calculation_logging(workdir, reset=True)
    LOG.info("Calculation started at %s; log: %s", datetime.now().isoformat(timespec="seconds"), log_path)
    run_simulation(work_dir=workdir, config_path=config_path)
    return 0


def _run_dataset(argv: list[str]) -> int:
    return zf.main(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args(argv)

    if args.subcommand == "dataset":
        return _run_dataset(unknown_args)

    config_path = args.config_file.resolve()
    assert config_path.exists(), f"Config file does not exist: {config_path}"
    workdir = _resolve_workdir(config_path, args.workdir)

    dispatch = {
        "build_model": _run_build_model,
        "simulate": _run_simulate,
    }
    return dispatch[args.subcommand](config_path=config_path, workdir=workdir)


if __name__ == "__main__":
    raise SystemExit(main())
