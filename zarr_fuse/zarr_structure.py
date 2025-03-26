import yaml
import attrs
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Sequence, Tuple


reserved_keys = set(['ATTRS', 'COORDS', 'VARS'])


@attrs.define
class Variable:
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None
    coords: Union[str, List[str]] = None
    df_col: Optional[str] = None

    def __attrs_post_init__(self):
        """Set df_cols to [name] if not explicitly provided."""
        if self.df_col is None:
            self.df_col = self.name
        if self.coords is None:
            self.coords = [self.name]
        if isinstance(self.coords, str):
            self.coords = [self.coords]

    @property
    def attrs(self):
        return dict(
            unit=self.unit,
            description=self.description,
            df_col=self.df_col
        )


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
class Coord:
    name: str
    description: Optional[str] = None
    composed: Dict[str, List[Any]] = None
    values: Optional[np.ndarray]  = None              # Fixed size, given coord values.
    chunk_size: Optional[int] = 1024    # Explicit chunk size, 1024 default. Equal to 'len(values)' for fixed size coord.

    def __attrs_post_init__(self):
        if self.composed is None:
            self.composed = [self.name]

        if self.values is None:
            self.values = [list() for _ in self.composed]
        elif isinstance(self.values, int):
            self.values = [np.arange(self.values)]
        elif isinstance(self.values, list):
            # Can express list of empty arrays (for tuple components) as [[], []]
            # but can not express empty list of pairs (given second dimension while first is 0.
            if len(self.values) == 0:
                self.values = [list() for _ in self.composed]
            else:
                self.values = list(zip(*self.values))
        self.values = np.atleast_2d(self.values)
        assert len(self.values) == len(self.composed)


Attrs = Dict[str, Any]
ZarrNodeStruc = Dict[str, Union[Attrs, Dict[str, Variable], Dict[str, Coord], 'ZarrNodeStruc']]

def set_name(d, name):
    d['name'] = name
    return d

def deserialize(content: 'stream') -> dict:
    """
    Recursively deserializes a dictionary.
    Processes special keys:
      - ATTRS: kept as is
      - COORDS: converted into a list of Coord objects
      - VARS: converted into a list of Quantity objects
    Other keys are processed recursively.
    """
    content = yaml.safe_load(content)
    result = {}
    if 'ATTRS' in content:
        result['ATTRS'] = content['ATTRS']
    if 'VARS' in content:
        result['VARS'] = {k: Variable(**set_name(v, k)) for k, v in content['VARS'].items()}
    if 'COORDS' in content:
        result['COORDS'] = {k: Coord(**set_name(c, k)) for k, c in content['COORDS'].items()}
    for key, value in content.items():
        if key not in ['ATTRS', 'COORDS', 'VARS']:
            result[key] = deserialize(value) if isinstance(value, dict) else value
    return result



def convert_value(obj):
    """
    Recursively convert an object for YAML serialization.

    - If obj is an instance of an attrs class, convert it to a dict using
      attrs.asdict with this function as the value_serializer.
    - If obj is a dict, list, or tuple, process its elements recursively.
    - For basic types (int, float, str, bool, None), return the value as is.
    - Otherwise, return the string representation of obj.
    """
    if attrs.has(obj):
        return attrs.asdict(obj, value_serializer=lambda inst, field, value: convert_value(value))
    elif isinstance(obj, dict):
        return {k: convert_value(v) for k, v in obj.items()}
    elif hasattr(obj, 'dtype'):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_value(item) for item in obj]
    else:
        return obj

def serialize(hierarchy: dict) -> str:
    """
    Serialize a hierarchy of dictionaries (and lists/tuples) with leaf values that
    may be instances of attrs classes to a YAML string.

    The conversion is performed by the merged convert_value function which uses a
    custom value serializer for attrs.asdict.
    """
    converted = convert_value(hierarchy)
    return yaml.safe_dump(converted, sort_keys=False)


def read_structure(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return deserialize(file)


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