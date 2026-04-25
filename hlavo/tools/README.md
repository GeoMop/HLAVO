# Auxiliary tools in HLAVO project

## list_zarr_fuse_units.py

Reads all units from given schema, checks validity and prints their names according to the lib.
User can check visually that units are represented as expected.
Can also print out all available units.

## zf.py

Lists the real content of HLAVO zarr stores declared in `hlavo/schemas`.
Supports one optional full-path glob like `wells.zarr/Uhelna/*`.
