# How to work with GRB data


## Dataset, Aladin numerical model, CZ only, step 1km
[FTP URL](https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/)
[Documentation](https://geoportal.gov.cz/php/micka/record/basic/664b4599-3354-43e6-a1e4-314efcc0a8017c)
Boundary: 12.09, 48.551, 18.860, 51.056

Summary of example GRB file:

Dataset summary:
<xarray.Dataset> Size: 42MB
Dimensions:            (step: 73, latitude: 290, longitude: 501)
Coordinates:
  * step               (step) timedelta64[ns] 584B 00:00:00 ... 3 days 00:00:00
    valid_time         (step) datetime64[ns] 584B 2026-02-08 ... 2026-02-11
  * latitude           (latitude) float64 2kB 48.5 48.51 48.52 ... 51.09 51.1
  * longitude          (longitude) float64 4kB 12.0 12.01 12.03 ... 18.98 19.0
    time               datetime64[ns] 8B 2026-02-08
    heightAboveGround  float64 8B 0.0
Data variables:
    unknown            (step, latitude, longitude) float32 42MB 600.9 ... 5.0...
Attributes:
    GRIB_edition:            1
    GRIB_centre:             89
    GRIB_centreDescription:  Prague
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             Prague
    history:                 2026-02-10T18:51 GRIB to CDM+CF via cfgrib-0.9.1...



## Full ALADIN mode, Europe
[FTP URL](https://opendata.chmi.cz/meteorology/weather/nwp_aladin/)
[Documentation](https://geoportal.gov.cz/php/micka/record/basic/664b1c63-c510-40b8-9a84-313264c0a8017c)
Boundary: 1.288, 36.038, 35.145, 55.253

