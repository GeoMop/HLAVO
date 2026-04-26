from importlib import import_module
from typing import Sequence


def resolve_named_class(class_name: str, module_names: Sequence[str]) -> type:
    assert class_name, "Configured class name must not be empty"
    for module_name in module_names:
        module = import_module(module_name)
        resolved = getattr(module, class_name, None)
        if isinstance(resolved, type):
            return resolved
    raise ValueError(f"Unable to resolve class '{class_name}' from modules: {', '.join(module_names)}")
