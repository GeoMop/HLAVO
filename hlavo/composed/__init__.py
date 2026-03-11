# DRAFT
"""Exports for composed."""

from importlib import import_module

__all__ = ["main", "setup_models", "Model1D", "Model3D"]


def __getattr__(name: str):
    if name == "main":
        return import_module(f"{__name__}.run_map").main
    if name == "setup_models":
        return import_module(f"{__name__}.composed_model_mock").setup_models
    if name == "Model1D":
        return import_module(f"{__name__}.model_1d_worker").Model1D
    if name == "Model3D":
        return import_module(f"{__name__}.model_3d_driver").Model3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
