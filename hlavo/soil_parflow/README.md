# Parflow model


TODO here:

- move here: parflow_model.py
- make a test that runs a single simulation

## Support for CLM model

- instalace Parflow v prostředích musí zahrnovat CLM modul
- otestovat použití [CLM](https://www.cesm.ucar.edu/models/clm) přímo v parflow
- [CLM sources](https://github.com/HPSCTerrSys/CLM3.5)
- PArflow CLM papers: 
    Maxwell, R.M. and N.L. Miller, Journal of Hydrometeorology 6(3):233-247, 2005
    Kollet, S.J. and R.M. Maxwell, Water Resources Research 44:W02402, 2008
- start with Parflow Python CLM example [clm.py](https://github.com/parflow/parflow/blob/master/test/python/clm/clm.py)
  and other in the same directory
    
## Input meteo data
See node 'chmi_aladin_10m' in [zarr-fuse schema](../ingress/scrapper/schemas/hlavo_surface_schema.yaml).


