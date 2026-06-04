#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hlavo.composed.model_composed import run_simulation
from hlavo.deep_model.build_modflow_grid import build_model
from hlavo.tools import zf

LOG = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(process)d %(name)s: %(message)s"


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


def configure_calculation_logging(workdir: Path, log_name: str = "calculation.log", reset: bool = False) -> Path:
    """
    Configure main-process calculation logging.

    The file handler is intended for the process that owns the calculation.
    Dask workers are not attached to this log file because concurrent worker
    writes would need explicit synchronization.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / log_name
    if reset:
        log_path.unlink(missing_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    resolved_log_path = log_path.resolve()

    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in root.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler._hlavo_calculation_handler = True
        root.addHandler(file_handler)

    for handler in list(root.handlers):
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and not getattr(handler, "_hlavo_stdout_handler", False)
        ):
            root.removeHandler(handler)

    has_stream_handler = any(getattr(handler, "_hlavo_stdout_handler", False) for handler in root.handlers)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        stream_handler._hlavo_stdout_handler = True
        root.addHandler(stream_handler)

    return log_path


def _run_build_model(*, config_path: Path, workdir: Path) -> int:
    build_config = build_model(config_source=config_path, workspace=workdir)
    LOG.info("3D model built in %s", build_config.workspace)
    return 0


def _run_simulate(*, config_path: Path, workdir: Path) -> int:
    log_path = configure_calculation_logging(workdir, reset=True)
    LOG.info("Calculation log: %s", log_path)
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
