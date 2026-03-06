# Parflow model




## Support for CLM model

- instalace Parflow v prostředích musí zahrnovat CLM modul
- otestovat použití [CLM](https://www.cesm.ucar.edu/models/clm) přímo v parflow
- [CLM sources](https://github.com/HPSCTerrSys/CLM3.5)
- PArflow CLM papers: 
    Maxwell, R.M. and N.L. Miller, Journal of Hydrometeorology 6(3):233-247, 2005
    Kollet, S.J. and R.M. Maxwell, Water Resources Research 44:W02402, 2008
- start with Parflow Python CLM example [clm.py](https://github.com/parflow/parflow/blob/master/test/python/clm/clm.py)
  and other in the same directory
   
## CLM models (JB, AI based)
- CLM current verion is 5.0
- Use either satellite phenology (prescribed seasonal variance) or biogeochemistry phenology (BP - modeled seasonal variance)
- For BP one can turn on /off crop growth model.
- phenology = seasonal variability
- these are build time switches !!, we have to check how to configure them during ParFlow build
- CLM build options: `-bgc [ none | cn | cndv ] -crop [ on | off ]`

### BP models (freely combined)
- CN - Carbon-Nitrogen 
- CNDV - active Carbon–Nitrogen with Dynamic Vegetation (non-managed plants) 
  legacy DGVM (Dynamic Global Vegetation Model); NOT SUPPORTED ANYMORE
- external FATES model for modern Dynamic Vegetaion model, not clear how to couple

- evergreen: leaves persist year-round; turnover still happens, but no strong leaf-off season
- seasonally deciduous: leaf timing responds mainly to day length
- stress deciduous: leaf timing responds mainly to temperature and/or soil moisture stress.

### Crop model (based on AgroIBIS model)
three phases:

- planting → leaf emergence
- leaf emergence → beginning of grain fill
- grain fill → physiological maturity and harvest.


## Input meteo data
See node 'chmi_aladin_10m' in [zarr-fuse schema](../ingress/scrapper/schemas/hlavo_surface_schema.yaml).

- Meteorological data: CLM needs DSWR, DLWR, APCP, Temp, UGRD, VGRD, Press, SPFH (see [https://parflow.readthedocs.io/en/latest/keys.html#clm-solver-parameters]);
  these can be computed from CHMI data:
                surface_solar_radiation_downwards
                surface_thermal_radiation_downwards
                precipitation_amount_accum
                air_temperature_2m
                wind_speed_10m
                wind_from_direction_10m
                air_pressure_at_sea_level
                relative_humidity_2m
  The CHMI time-dependent data are passed to Toyroblem.run() as xarray.Dataset.

## Stationary parameters, instant processes control

- Parameters controling short term processes (temperature, moisture, light, ..) not seasonal changes
- Vegetation data (LAI, SAI, Z0M, DISPLA, ... ): defined in `drv_vegp.dat`;
  these data are fixed for given soil type and must be adjusted to seasonality.
  It is possible to use time- and space-dependent vegetation, but I think it makes sense
  only in long-term simulations.
  TODO: description which types we will use and what parameter value ranges we will consider.

## Seasonal variability control (phenology)
