#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from hlavo.composed.composed_model_mock import run_simulation
from hlavo.deep_model.build_modflow_grid import build_model

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


def _run_build_model(*, config_path: Path, workdir: Path) -> int:
    config_source = _translate_build_model_config(config_path, workdir)
    build_config = build_model(config_source=config_source, workspace=workdir)
    LOG.info("3D model built in %s", build_config.workspace)
    return 0


def _run_simulate(*, config_path: Path, workdir: Path) -> int:
    run_simulation(work_dir=workdir, config_path=config_path)
    return 0


def _translate_build_model_config(config_path: Path, workdir: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    assert isinstance(raw, dict), "Config YAML must be a mapping"
    if "model" in raw:
        return raw

    model_3d = raw["model_3d"]
    assert isinstance(model_3d, dict), "model_3d config must be a mapping"

    translated = dict(raw)
    translated["_config_path"] = str(config_path)
    translated["model"] = {
        "model_name": "model_with_mine",
        "workspace": str(workdir),
        "sim_name": str(model_3d.get("name", "uhelna")),
        "exe_name": str(model_3d.get("executable", "mf6")),
        "simulation_days": float(model_3d.get("total_time_days", 1.0)),
        "drain_conductance": float(model_3d.get("drain_conductance", 1.0)),
    }
    return translated


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

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
