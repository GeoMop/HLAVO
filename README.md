---
![Acknowledgement](doc/graphics/ack.svg)
---

# HLAVO
System for numerical prediction of water table using continuously collected meteorological and 
soil mosture profile data.

## Get started
The HLAVO system deployment needs to setup three kind of access:

1. Access to to associated DVC storage with site specific large input files.
   Follow the instuctions in [DVC usage doc](doc/dvc_usage.md).
   
2. Access to zarr_fuse storage, through setting secrets in `.secrets_env`, 
   see [HLAVO secretes document](https://docs.google.com/document/d/1uNnEJvaM2AmjSlnt9kfAbJh9ZcPZBLFXIjh6WPCNdOg/edit?usp=sharing)

3. Deployment of active ingress and dashboard services. Done through github actions 
   AGNET: review github actions and which github secrets they use, refer to general zarr_fuse doc for ingress and dashboard.

## Data ingress
HLAVO zarr-fuse schemas are indexed in `hlavo/schemas` as symlinks to the
ingress package that owns each dataset. The update procedures are:

- `hlavo/schemas/profile_schema.yaml` -> `hlavo/ingress/moist_profile/profile_schema.yaml`
  - Owned by the moisture profile pipeline.
  - Refresh raw DataFlow exports with `hlavo/ingress/moist_profile/dataflow_grab.py`.
  - Update site metadata CSV files in `hlavo/ingress/moist_profile/extract/`
    from the project spreadsheet; use `gpx2csv.py` only to regenerate waypoint
    CSV support data from `export.gpx`.
  - Write profiles and lab profile data with
    `hlavo.ingress.moist_profile.extract.main.run_extract(...)` or the script's
    `main(source_dir, storage_path)` wrapper.

- `hlavo/schemas/wells_schema.yaml` -> `hlavo/ingress/well_data/wells_schema.yaml`
  - Owned by the well water-level and draw pipeline.
  - Pull or update the source XLSX files, then run
    `cd hlavo/ingress/well_data && ../../../dev/hlavo run python well_data_process.py`.
  - Add `plot` to the command to also generate the local PDF review output.

- `hlavo/schemas/chmi_stations_schema.yaml` ->
  `hlavo/ingress/meteo_playground/chmi_stations/chmi_stations_schema.yaml`
  - Owned by the CHMI station/Open-Meteo processing scripts.
  - Refresh CHMI metadata outputs with `meta_description.py` and
    `meta_processing.py` when station or quantity metadata changes.
  - Download raw CHMI/Open-Meteo data with `data_scrapper.py`.
  - Update the zarr-fuse nodes and ParFlow/CLM forcing node with
    `data_processing.py`.

- `hlavo/schemas/hlavo_surface_schema.yaml` ->
  `hlavo/ingress/scrapper/schemas/hlavo_surface_schema.yaml`
  - Owned by the automatic ingress service, not by a one-shot local update
    script.
  - The service configuration is `hlavo/ingress/scrapper/endpoints_config.yaml`.
    It connects active endpoint definitions to extractor functions under
    `hlavo/ingress/scrapper/extract/`.
  - Current extractors normalize yr.no JSON forecasts and CHMI ALADIN 1 km GRIB
    files before the ingress worker writes the configured target zarr-fuse node.


## Simulation workflow
TBD.

---

# Developers corner


# Kalman

## Local run
Use docker image: `martinspetlik/kalman_parflow:v1.0.0`

```bash
docker run --rm -it -v HLAVO_repository:/HLAVO martinspetlik/kalman_parflow:v1.0.0 python3.10 /HLAVO/soil_model/kalman.py work_dir /HLAVO/soil_model/configs/case_parameter_inv_synth.yaml
```


## Charon cluster run
First singularity image has to be created:
```bash
export SINGULARITY_CACHEDIR="user home dir"
export SINGULARITY_LOCALCACHEDIR="user scratch dir"
export SINGULARITY_TMPDIR=$SCRATCHDIR

singularity build kalman_parflow.sif docker://martinspetlik/kalman_parflow:v1.0.0
```

Run Kalman on a directory that contains a configuration file
```bash
./charon_run_kalman.sh directory
```

## DVC management of large files
Do not upload large binary files (over few MB) to the git repository, there are tight limits of the
git technology and GitHub hosting to 500MB for the whole repository including all history. Binary files are not stored as diffs
so they bite large chunks of the available space.
Use DVC instaed to automate upload of large files to Google Drive. See [detailed instructions](doc/dvc_usage.md).

## Surface model (`surface_model` dir)
Model of surface infiltration layer. Composed of Richards model and Kalman filter
for assimilation of the soil moisture profile measurement.

#- [code](./soil_model/README.md)
#- [run configurations](./soil_model/config/README.md)

## Deep Vadose Zone model (`deep_model`)

subfolder `GIS` is for various GIS resources.
