from __future__ import annotations

import logging
import sys
from pathlib import Path

HLAVO_ROOT_LOGGER = logging.getLogger()
HLAVO_DEBUG_LOGGER = logging.getLogger("hlavo")

# Keep INFO readable on stdout while preserving source details only for DEBUG diagnostics.
INFO_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
DEBUG_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(process)5d %(name)s:%(lineno)d: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


class LevelFormatter(logging.Formatter):
    """Use compact operator INFO lines and source-aware DEBUG diagnostics."""

    def __init__(self):
        super().__init__(datefmt=LOG_DATE_FORMAT)
        self._info_formatter = logging.Formatter(INFO_LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        self._debug_formatter = logging.Formatter(DEBUG_LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.DEBUG:
            return self._debug_formatter.format(record)
        return self._info_formatter.format(record)


def set_hlavo_loggers() -> None:
    """Apply the base HLAVO logger policy.

    Specification:
    - `HLAVO_ROOT_LOGGER` stays at INFO, so third-party libraries remain quiet
    - `HLAVO_DEBUG_LOGGER` is raised to DEBUG, so HLAVO diagnostics can still
      reach DEBUG-capable handlers attached to that logger tree

    Usage:
    call this before installing stdout or file handlers managed by this module.
    """
    HLAVO_ROOT_LOGGER.setLevel(logging.INFO)
    # Keep dependency logs at INFO while allowing HLAVO DEBUG diagnostics into file handlers.
    HLAVO_DEBUG_LOGGER.setLevel(logging.DEBUG)


def ensure_stdout_handler(flag_attr: str) -> None:
    """Install the canonical HLAVO stdout handler on `HLAVO_ROOT_LOGGER`.

    Specification:
    - removes pre-existing non-file stream handlers not marked by `flag_attr`
    - installs exactly one stdout stream handler marked by `flag_attr`
    - emits INFO and above using the standard HLAVO formatter

    Usage:
    use a caller-specific `flag_attr` value so repeated setup stays idempotent.
    """
    formatter = LevelFormatter()

    for handler in list(HLAVO_ROOT_LOGGER.handlers):
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and not getattr(handler, flag_attr, False)
        ):
            HLAVO_ROOT_LOGGER.removeHandler(handler)

    has_stream_handler = any(getattr(handler, flag_attr, False) for handler in HLAVO_ROOT_LOGGER.handlers)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, flag_attr, True)
        HLAVO_ROOT_LOGGER.addHandler(stream_handler)


def ensure_debug_file_handler(log_path: Path, flag_attr: str, *, filter_: logging.Filter | None = None) -> Path:
    """Install one DEBUG-capable file handler for a log path on `HLAVO_DEBUG_LOGGER`.

    Specification:
    - keeps setup idempotent for the `(HLAVO_DEBUG_LOGGER, resolved_log_path, flag_attr)` tuple
    - uses the standard HLAVO formatter
    - applies the optional `filter_` only when creating a new handler

    Usage:
    use for HLAVO-only files, for example the main debug log or per-worker DEBUG logs.
    """
    formatter = LevelFormatter()
    resolved_log_path = Path(log_path).resolve()

    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        and getattr(handler, flag_attr, False)
        for handler in HLAVO_DEBUG_LOGGER.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(resolved_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        if filter_ is not None:
            file_handler.addFilter(filter_)
        setattr(file_handler, flag_attr, True)
        HLAVO_DEBUG_LOGGER.addHandler(file_handler)

    return resolved_log_path
