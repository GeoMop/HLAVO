# Model1D – Usage Guide

## Overview

`hlavo.composed.model_1d.Model1D` is a one-dimensional hydrological model component designed to operate within a coupled 1D–3D simulations. It integrates:

* Measurement data
* Meteorological data
* An Unscented Kalman filter (UKF)

The model processes time-stepped data and communicates with external components via queues.

## Testing

Run tests using:

```bash
pytest tests/model_1d/model_1d_test_run.py
```

Or manually:

```bash
python tests/model_1d/model_1d_test_run.py
```

### Covered tests

* Measurement preparation
* Time slicing
* Coordinate (lon, lat) retrieval
* Step execution

---

## Known Limitations / TODOs

* Meteorological data is currently mocked
* Time alignment hack (forcing year = 2025)
* Measurement covariance matrix is simplistic (identity)

---



## Charon cluster run - not tested
First singularity image has to be created:
```bash
export SINGULARITY_CACHEDIR="user home dir"
export SINGULARITY_LOCALCACHEDIR="user scratch dir"
export SINGULARITY_TMPDIR=$SCRATCHDIR

singularity build hlavo_0_1_0.sif  docker://flow123d/hlavo:0.1.0


./pbs_run.sh
