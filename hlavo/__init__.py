# DRAFT
"""Top-level HLAVO package exports with lazy submodule loading."""

from importlib import import_module

__all__ = ["soil_py", "soil_parflow", "kalman", "deep_model", "composed"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
