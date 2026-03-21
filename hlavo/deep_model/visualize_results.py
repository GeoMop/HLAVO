from __future__ import annotations

import argparse
import logging
from pathlib import Path

from create_paraview import create_paraview
from create_plots import create_plots

LOG = logging.getLogger(__name__)


def visualize_results(config_path: Path, workspace: Path | None = None) -> None:
    create_paraview(config_path, workspace)
    create_plots(config_path, workspace)
    LOG.info("Visualization complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Paraview outputs and plots from finished Modflow run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override model workspace directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    visualize_results(args.config, args.workspace)


if __name__ == "__main__":
    main()
