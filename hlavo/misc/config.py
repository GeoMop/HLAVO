from __future__ import annotations

from pathlib import Path

import yaml


def _extract_mapping(raw: dict, path: tuple[str, ...]) -> dict | None:
    current = raw
    for key in path:
        if key not in current:
            return None
        current = current[key]
        if not isinstance(current, dict):
            raise AssertionError(".".join(path) + " must be a mapping")
    return current


def load_config(
    config_source: Path | dict,
    path: tuple[str, ...] | None = None,
) -> tuple[dict, Path | None]:
    if isinstance(config_source, dict):
        raw = config_source
        config_path_raw = raw.get("_config_path")
        config_path = None if config_path_raw is None else Path(str(config_path_raw))
    else:
        config_path = Path(config_source).resolve()
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    assert isinstance(raw, dict), "Config YAML must be a mapping"
    if config_path is not None:
        raw = dict(raw)
        raw["_config_path"] = str(config_path)
    if path is None:
        return raw, config_path
    selected = _extract_mapping(raw, path)
    assert selected is not None, "Missing required config section: " + ".".join(path)
    if config_path is not None and selected.get("_config_path") != str(config_path):
        selected = dict(selected)
        selected["_config_path"] = str(config_path)
    return selected, config_path


def require_mapping(raw: dict, key: str) -> dict:
    value = raw[key]
    assert isinstance(value, dict), f"{key} must be a mapping"
    return value


def optional_mapping(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    assert isinstance(value, dict), f"{key} must be a mapping"
    return value


def to_float(raw: dict, key: str, default: float | None = None) -> float:
    if key in raw:
        return float(raw[key])
    assert default is not None, f"{key} is required"
    return float(default)


def to_optional_float(raw: dict, key: str) -> float | None:
    if key not in raw:
        return None
    return float(raw[key])


def to_optional_int(raw: dict, key: str) -> int | None:
    if key not in raw:
        return None
    return int(raw[key])


# AGENT: no default path,
# rename to get_path
def get_path(raw: dict, key: str) -> Path:
    return Path(str(raw[key]))
