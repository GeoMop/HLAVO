# HLAVO
System for numerical prediction of water table using continuously collected meteorological and 
soil mosture profile data.




---
# Developers corner

## Kalman
# Local run
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

