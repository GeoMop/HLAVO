# Zarr-fuse datasets

## Zarr-fuse overview
[zarr-fuse](https://github.com/GeoMop/zarr_fuse) is a Python package and set of web services
for schema based organization of large geospatial datasets in Zarr format. 
The structure of datasets is defined by a schema written in YAML, allowing to document individual variables
their textual description, units and resources for the data provenance.
ZARR format itself was choosen for ability of parallel writes and reads, chunking of large arrays and 
flexible operations through xarray and dask libraries.

## Meteo datasets
Several meteorological datasets are available through zarr-fuse automatic data ingerssion pipelines.
These are used by the surface models to predict water infiltration. These are fussed into a common
dataset `meteo` before simulation of a particular period and then directly used for the evapo-transpiration model.


### `yr.no` dataset

## Surface profile measurements

## Water table measurements

## Simulations
