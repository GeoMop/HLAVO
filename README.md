# HLAVO
Software part of the HLAVO project.

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
