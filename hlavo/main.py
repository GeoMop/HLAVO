#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary HLAVO mock entrypoint for launcher validation."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="mock",
        help="Mock command name to report back in logs.",
    )
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the mock command.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    logging.getLogger("hlavo.main").info(
        "mock entrypoint invoked",
        extra={},
    )
    logging.info("command: %s", args.command)
    logging.info("command args: %s", args.command_args)
    logging.info("working directory: %s", Path.cwd())
    logging.info("python executable: %s", sys.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
