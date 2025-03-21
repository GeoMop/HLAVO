import yaml
from zarr_structure import read_structure, read_storage

"""
This is an inital test of xarray, zarr functionality that we build on.
This requires dask.
"""
def aux_read_struc(tmp_path, struc_dict):
    struc_path = tmp_path / "structure.yaml"
    with open(struc_path, "w") as f:
        f.write(struc_dict)

    structure = read_structure(struc_path)
    assert set(['COORDS', 'VARS', 'ATTRS']).issubset(set(structure.keys()))

    tree = read_storage(struc_path)
    return structure, tree

def test_read_structure_weather(tmp_path):
    # Example YAML file content (as a string for illustration):
    example_yaml = """
    COORDS:
      - name: "time"
        unit: "seconds"
        description: "Time coordinate"
        df_cols: ["timestamp"]
        chunk_size: 1024
      - name: "lat"
        unit: "degrees"
        description: "Latitude coordinate"
        df_cols: ["latitude"]
        chunk_size: 512

    VARS:
      - name: "temperature"
        unit: "K"
        description: "Temperature in Kelvin"
        df_cols: ["temp"]
        coords: ["time", "lat"]
    ATTRS:
        description: "Example dataset"
        url: "https://example.com"
    """
    structure, tree = aux_read_struc(tmp_path, example_yaml)
    assert len(structure['COORDS']) == 2
    assert len(structure['VARS']) == 1
    print("Coordinates:")
    for coord in structure["COORDS"]:
        print(coord)
    print("\nQuantities:")
    for var in structure["VARS"]:
        print(var)


def test_read_structure_tensors(tmp_path):
    # Example YAML file content (as a string for illustration):
    example_yaml = """
    COORDS:
      - name: "time"
        unit: TimeStamp[CET]
        description: "Time coordinate"
        df_cols: ["timestamp"]
        chunk_size: 256
      - name: elements
        unit: Index
        description: "Mesh element index"
        df_cols: ["element"]
        chunk_size: 512
      - name: tn_voigt_3d
        unit: Index
        size: 6

    VARS:
      - name: "pressure"
        unit: "m"
        description: "Pressure head field"
        df_cols: ["pressure"]
        coords: ["time", "elements"]
      - name: "pressure"
        unit: "m"
        description: "Pressure head field"
        df_cols: ["pressure"]
        coords: ["time", "elements"]        
    ATTRS:
        description: "Flow123d simulation result."
        flow123d_release: "3.9.0"
    """
    structure, tree = aux_read_struc(tmp_path, example_yaml)
    assert len(structure['COORDS']) == 3
    assert len(structure['VARS']) == 2
    print("Coordinates:")
    for coord in structure["COORDS"]:
        print(coord)
    print("\nQuantities:")
    for var in structure["VARS"]:
        print(var)
