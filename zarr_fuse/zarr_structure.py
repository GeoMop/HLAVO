import yaml
import attrs
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Sequence

import xarray as xr

reserved_keys = set(['ATTRS', 'COORDS', 'VARS'])

def df_cols_converter(value: Optional[Union[str, List[str]]], default:str) -> Optional[List[str]]:
    if value is None:
        return (default,)
    elif isinstance(value, str):
        return (value,)
    else:
        return tuple(value)

def coord_values_converter(values):
    if values is None:
        return np.array([])
    elif isinstance(values, int):
        size = values
        return np.arange(size)
    else:
        values = np.atleast_1d(values)
        assert values.ndim == 1
        return values


@attrs.define
class Array:
    name: str
    df_cols: Optional[Sequence[str]] = None
    unit: Optional[str] = None
    description: Optional[str] = None

    def __attrs_post_init__(self):
        """Set df_cols to [name] if not explicitly provided."""
        self.df_cols = df_cols_converter(self.df_cols, default=self.name)


def coords_converter(coords):
    if isinstance(coords, str):
        return [coords]
    else:
        return coords

@attrs.define
class Quantity(Array):
    coords: Union[str, List[str]] = attrs.field(converter=coords_converter, default=list)


@attrs.define
class Coord(Array):
    values: Optional[np.ndarray]  = attrs.field(converter=coord_values_converter, default=None)          # Fixed size, given coord values.
    chunk_size: Optional[int] = 1024    # Explicit chunk size, 1024 default. Equal to 'len(values)' for fixed size coord.

Attrs = Dict[str, Any]
ZarrNodeStruc = Dict[str, Union[Attrs, List[Coord], List[Quantity], 'ZarrNodeStruc']]


def deserialize(content: dict) -> dict:
    """
    Recursively deserializes a dictionary.
    Processes special keys:
      - ATTRS: kept as is
      - COORDS: converted into a list of Coord objects
      - VARS: converted into a list of Quantity objects
    Other keys are processed recursively.
    """
    result = {}
    if 'ATTRS' in content:
        result['ATTRS'] = content['ATTRS']
    if 'COORDS' in content:
        result['COORDS'] = [Coord(**c) for c in content['COORDS']]
    if 'VARS' in content:
        result['VARS'] = [Quantity(**v) for v in content['VARS']]
    for key, value in content.items():
        if key not in ['ATTRS', 'COORDS', 'VARS']:
            result[key] = deserialize(value) if isinstance(value, dict) else value
    return result


def serialize(structure: dict) -> dict:
    """
    Recursively serializes a structure.
    Converts any attrs-decorated objects to dictionaries.
    """
    result = {}
    for key, value in structure.items():
        if isinstance(value, dict):
            result[key] = serialize(value)
        elif isinstance(value, list):
            result[key] = [attrs.asdict(v) if attrs.has(v) else v for v in value]
        else:
            result[key] = value
    return result


def read_structure(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        structure_dict = yaml.safe_load(file)
    return deserialize(structure_dict)


def write_structure(structure: dict, yaml_path: str) -> None:
    serialized_structure = serialize(structure)
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(serialized_structure, file)

# Example Usage:
# tree_structure = read_structure('structure.yaml')
# write_structure(tree_structure, 'output_structure.yaml')

# def build_xarray_tree(structure: dict, source_path: Path = None) -> xr.DataTree:
#     """
#     Recursively builds an xarray DataTree from the given structure.
#
#     For each node:
#       - Create an xarray.Dataset.
#       - Update global attributes from 'ATTRS' if available.
#       - Add coordinates from 'COORDS' (empty arrays).
#       - Add variables from 'VARS' as DataArrays with dummy data.
#
#     Child nodes (any keys not in ['ATTRS', 'COORDS', 'VARS'])
#     are recursively processed and attached to the DataTree.
#
#     Parameters:
#       structure (dict): The deserialized structure.
#       name (str): Name for the current DataTree node.
#
#     Returns:
#       xr.DataTree: The resulting DataTree.
#     """
#     ds = xr.Dataset()
#
#     # Set dataset attributes if provided.
#     if "ATTRS" in structure:
#         ds.attrs.update(structure["ATTRS"])
#
#     # Process coordinates.
#     for coord in structure.get("COORDS", []):
#         # Add coordinate as an empty array; in practice you may want to fill real data.
#         ds = ds.assign_coords({coord.name: []})
#
#     # Process variables.
#     for var in structure.get("VARS", []):
#         if getattr(var, "shape", None) and getattr(var, "coords", None):
#             # Create a dummy zeros array using the provided shape and assign dims.
#             data = np.zeros(var.shape)
#             ds[var.name] = xr.DataArray(data, dims=var.coords)
#         else:
#             # Fallback to a scalar zero if no shape/dims info is provided.
#             ds[var.name] = xr.DataArray(0)
#
#     # Recursively build children DataTree nodes.
#     children = {}
#     for key, value in structure.items():
#         if key not in ['ATTRS', 'COORDS', 'VARS']:
#             children[key] = build_xarray_tree(value, name=key)
#
#     # Create and return the DataTree node.
#     if source_path:
#         name = source_path.name
#     else:
#         name = "."
#     ds.attrs["source_path"] = str(source_path)
#     return xr.DataTree(ds, name=name, children=children)

# def read_storage(yaml_path: Path) -> xr.DataTree:
#     structure = read_structure(yaml_path)
#     return build_xarray_tree(structure, source_path=yaml_path)