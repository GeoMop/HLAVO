from typing import Sequence


def resolve_named_class(class_name: str, classes: Sequence[type]) -> type:
    assert class_name, "Configured class name must not be empty"
    for cls in classes:
        assert isinstance(cls, type), f"Class resolver item must be a class, got: {type(cls)}"
        if cls.__name__ == class_name:
            return cls
    allowed_names = ", ".join(cls.__name__ for cls in classes)
    raise ValueError(f"Unable to resolve class '{class_name}' from classes: {allowed_names}")
