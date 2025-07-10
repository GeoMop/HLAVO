# HLAVO
Sowftware part of the HLAVO project.

# Local run
Use docker image: `martinspetlik/kalman_parflow:v1.0.0`

```bash
docker run --rm -it -v HLAVO_repository:/HLAVO martinspetlik/kalman_parflow:v1.0.0 python3.10 /HLAVO/soil_model/kalman.py work_dir /HLAVO/soil_model/configs/case_parameter_inv_synth.yaml
```


# Charon run
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


