import numpy as np
import xarray as xr

# Create first dataset with a coordinate "resolution" of size 2.
ds1 = xr.Dataset(
    {
        "data": (("x"), np.array([1, 2]))
    },
    coords={
        "x": [0, 1],
    }
)

# Create second dataset with a coordinate "resolution" of size 3.
ds2 = xr.Dataset(
    {
        "data": (("x"), np.array([5, 6, 7, 8]))
    },
    coords={
        "x": [0, 1, 2, 3],
    }
)

# Build a DataTree with ds1 as the root and ds2 as a child node.
dt = xr.DataTree(ds1, children={"child": xr.DataTree(ds2)}, name="root")

# Print the global coordinates view.
print("Global coordinates (dt.coords):")
print(dt.coords)
print("\nGlobal dataset view (dt.dataset):")
print(dt.dataset)
