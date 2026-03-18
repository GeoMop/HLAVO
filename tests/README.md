# Unit tests of key parts of the HLAVO system

## Test data
- `ingress/moist_profile/test_storage` 
   small test profile data just to operate on the same data structure
   schema file: `../hlavo/ingress/moist_profile/profile_schema.yaml`
   profile dataset under node `Uhelna/profiles`
   date_time coordinate does not match profile dataset, we need an optional parameter to shift the 
   date_time coordinates to given start date_time.
   
- `ingress/scrapper/test_meteo_storage`
   small meteo dataset with 
   schema file: `../hlavo/ingress/scrapper/schemas/hlavo_surface_schema.yaml`
   chmi meteo dataset under node `chmi_aladin_10m`
