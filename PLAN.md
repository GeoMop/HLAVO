## Curent goals


## Current Repository State

- `dashboard` - deployed, basic review functionality works fine
- `doc` - needs update
- `hlavo/ingress` - works, but missing processing of the detailed weather data
- `hlavo/composed` - need to finish refactoring
- `hlavo/deep_model` - finish refactoring to an implementation of the 3D model, not just standalone tool
- `hlavo/kalman` - need some reactoring to support both field simulations and laboratory simulations


## TODO points

### ToyProblem tests
- Resolved `2026-06-03`: documented `ToyProblem` input and output file policy, config parameters, and single-run parametrization in `hlavo/soil_parflow/parflow_model.py`. Verified with `python3 -m py_compile hlavo/soil_parflow/parflow_model.py tests/soil_parflow/test_parflow_model.py`.

### Fix infine loop in runs/composed_1d_only
Resolved `2026-06-03`: added main-process calculation logging and time-progress guards for the composed/Kalman run. `hlavo.main` writes `calculation.log` under the workdir and keeps INFO progress on stdout. Dask workers do not configure this global file log; key 1D-3D and Kalman time iterations are INFO, while array and detailed diagnostics are DEBUG by default. Worker exceptions are logged by the main process when futures re-raise. Non-advancing 1D/3D target times assert early. Verified with compile and diff checks; `bash runs/run_0.sh simulate runs/composed_1d_only/composed_config.yaml -w runs/composed_1d_only` is blocked in this environment by missing `dask`.

Original issue:
It seems there is an infinite loop somewhere in the Kalman.
Introduce a single log for whole calculation and log start of individual parflow subprocesses and
debugging array output. Use logging but with presence on the stdout for the Kalman time iterations.
We should know the time info with respect to the global time (from [3D]).

current stdout before Ctrl-C:
```
INFO hlavo.composed.model_composed: [SETUP] Submitted Model1D site_id=1
INFO hlavo.composed.model_3d: [3D] === Step: t=2025-03-06T00:00:00 -> t=2025-03-07T00:00:00 ===
INFO hlavo.composed.model_3d: [3D] send head -> 1D 0: date_time=2025-03-07T00:00:00, head=0.0
ukf.Q.shape  (16, 16)
ukf.Q  [[7.84000000e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 7.84000000e-02 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 7.84000000e-02 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.84000000e-02
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  7.84000000e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 7.84000000e-02 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 7.84000000e-02 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 7.84000000e-02
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  7.84000000e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 7.84000000e-02 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 7.84000000e-02 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 7.84000000e-02
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  7.84000000e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 7.84000000e-02 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 7.84000000e-02 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 5.33149243e-07]]
diag ukf.Q  [7.84000000e-02 7.84000000e-02 7.84000000e-02 7.84000000e-02
 7.84000000e-02 7.84000000e-02 7.84000000e-02 7.84000000e-02
 7.84000000e-02 7.84000000e-02 7.84000000e-02 7.84000000e-02
 7.84000000e-02 7.84000000e-02 7.84000000e-02 5.33149243e-07]
R measurement_noise_covariance  [[0.01 0.   0.   0.   0.  ]
 [0.   0.01 0.   0.   0.  ]
 [0.   0.   0.01 0.   0.  ]
 [0.   0.   0.   0.01 0.  ]
 [0.   0.   0.   0.   0.01]]
self.model  <hlavo.soil_parflow.parflow_model.ToyProblem object at 0x7643e8a75410>
data pressure  [-100.          -95.71428571  -91.42857143  -87.14285714  -82.85714286
  -78.57142857  -74.28571429  -70.          -65.71428571  -61.42857143
  -57.14285714  -52.85714286  -48.57142857  -44.28571429  -40.        ]
value: [-100.          -95.71428571  -91.42857143  -87.14285714  -82.85714286
  -78.57142857  -74.28571429  -70.          -65.71428571  -61.42857143
  -57.14285714  -52.85714286  -48.57142857  -44.28571429  -40.        ], noise: 0.004691122999071863, value + noise: [-99.99530888 -95.70959459 -91.42388031 -87.13816602 -82.85245173
 -78.56673745 -74.28102316 -69.99530888 -65.70959459 -61.42388031
 -57.13816602 -52.85245173 -48.56673745 -44.28102316 -39.99530888]
data array  [-99.99530888 -95.70959459 -91.42388031 -87.13816602 -82.85245173
 -78.56673745 -74.28102316 -69.99530888 -65.70959459 -61.42388031
 -57.13816602 -52.85245173 -48.56673745 -44.28102316 -39.99530888]
value: [-2.60775248], noise: -0.0028286334432866328, value + noise: [-2.61058111]
data array  [-2.61058111]
init cov  (16, 16)
np.diag(init_cov)  [4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00
 4.e+00 4.e+00 4.e+00 4.e+00 4.e+00 1.e-04]
[UKF] Running Kalman step (pid=62)
[UKF] Step 1: 2025-03-06T00:00:00.000000000 → 2025-03-06T01:00:00.000000000
meteo_ds  <xarray.Dataset> Size: 420B
Dimensions:               (date_time: 2)
Coordinates:
  * date_time             (date_time) datetime64[ns] 16B 2025-03-06 2025-03-0...
    site_id               int32 4B 1
Data variables: (12/18)
    APCP                  (date_time) float64 16B 0.0 0.0
    D10_source_station    (date_time) StringDType() 32B '0-203-0-20407036001'...
    DLWR                  (date_time) float64 16B 163.3 165.1
    DSWR                  (date_time) float64 16B 0.0 0.0
    F_source_station      (date_time) StringDType() 32B '0-203-0-11601' '0-20...
    H_source_station      (date_time) StringDType() 32B '0-203-0-11601' '0-20...
    ...                    ...
    Temp                  (date_time) float64 16B 277.2 277.8
    UGRD                  (date_time) float64 16B -0.1152 -0.1727
    VGRD                  (date_time) float64 16B 3.298 3.295
    elevation             (date_time) float64 16B 303.0 303.0
    latitude              (date_time) float64 16B 50.86 50.86
    longitude             (date_time) float64 16B 14.89 14.89
Attributes:
    description:       This is the ultimate first version of meteorologic dat...
    VERSION:           0.0.0
    station_priority:  ['0-203-0-20407036001', '0-203-0-11601', '0-20000-0-11...
    notes:             Preliminary CHMI-to-ParFlow/CLM forcing conversion. DS...
    time_step:         3600000000000
    time_interval:     189385200000000000
    __structure__:     ATTRS:\n  description: This is the ultimate first vers...
sqrt func call
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
[UKF] R shape: (5, 5); measurement shape: (5,)
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07351246]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07351117]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564865 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07565688 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07723025 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721649 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.07940825 0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08083931 0.07939993 0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08083577 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350386]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350515]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.0756451  0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07563689 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07720075 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721448 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793776  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08080844 0.07938589 0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08081197 0.0793929  0.07721548 0.07564687 0.07350816]
measurement function output size =   [0.08082385 0.0793929  0.07721548 0.07564687 0.07350816]
[UKF] Covariance sum: 57.18015019009751
[UKF] State estimate: [-14.0408214  -95.706343   -91.42388005 -86.40807311 -82.34140958
 -78.56673708 -74.28102279 -69.81713847 -64.70585612 -61.42387811
 -54.4976054  -52.67249326 -45.72760018 -38.58343416 -36.60448493
  -2.61058111]
[UKF] Step 2: 2025-03-06T01:00:00.000000000 → 2025-03-06T02:00:00.000000000
meteo_ds  <xarray.Dataset> Size: 420B
Dimensions:               (date_time: 2)
Coordinates:
  * date_time             (date_time) datetime64[ns] 16B 2025-03-06T01:00:00 ...
    site_id               int32 4B 1
Data variables: (12/18)
    APCP                  (date_time) float64 16B 0.0 0.0
    D10_source_station    (date_time) StringDType() 32B '0-203-0-20407036001'...
    DLWR                  (date_time) float64 16B 165.1 165.5
    DSWR                  (date_time) float64 16B 0.0 0.0
    F_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    H_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    ...                    ...
    Temp                  (date_time) float64 16B 277.8 277.3
    UGRD                  (date_time) float64 16B -0.1727 0.1187
    VGRD                  (date_time) float64 16B 3.295 3.398
    elevation             (date_time) float64 16B 303.0 303.0
    latitude              (date_time) float64 16B 50.86 50.86
    longitude             (date_time) float64 16B 14.89 14.89
Attributes:
    description:       This is the ultimate first version of meteorologic dat...
    VERSION:           0.0.0
    station_priority:  ['0-203-0-20407036001', '0-203-0-11601', '0-20000-0-11...
    notes:             Preliminary CHMI-to-ParFlow/CLM forcing conversion. DS...
    time_step:         3600000000000
    time_interval:     189385200000000000
    __structure__:     ATTRS:\n  description: This is the ultimate first vers...
sqrt func call
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
[UKF] R shape: (5, 5); measurement shape: (5,)
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.07356972]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.07356838]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577595 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07578455 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07772778 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.0777125  0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08052076 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08259049 0.08051259 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08258448 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.07356089]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.07356223]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577235 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07576377 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07769522 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771046 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08048608 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08255007 0.08049422 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08255607 0.08050339 0.07771148 0.07577414 0.0735653 ]
measurement function output size =   [0.08257024 0.08050339 0.07771148 0.07577414 0.0735653 ]
[UKF] Covariance sum: 58.27819502128289
[UKF] State estimate: [-10.28522876 -95.68377582 -91.4238796  -85.65174742 -81.81422763
 -78.56673658 -74.28102232 -69.63460138 -63.65227583 -61.4238726
 -51.55544947 -52.48792146 -42.48462026 -31.05276283 -32.51979321
  -2.61058111]
[UKF] Step 3: 2025-03-06T02:00:00.000000000 → 2025-03-06T03:00:00.000000000
meteo_ds  <xarray.Dataset> Size: 420B
Dimensions:               (date_time: 2)
Coordinates:
  * date_time             (date_time) datetime64[ns] 16B 2025-03-06T02:00:00 ...
    site_id               int32 4B 1
Data variables: (12/18)
    APCP                  (date_time) float64 16B 0.0 0.0
    D10_source_station    (date_time) StringDType() 32B '0-203-0-20407036001'...
    DLWR                  (date_time) float64 16B 165.5 165.2
    DSWR                  (date_time) float64 16B 0.0 0.0
    F_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    H_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    ...                    ...
    Temp                  (date_time) float64 16B 277.3 276.8
    UGRD                  (date_time) float64 16B 0.1187 0.05585
    VGRD                  (date_time) float64 16B 3.398 3.2
    elevation             (date_time) float64 16B 303.0 303.0
    latitude              (date_time) float64 16B 50.86 50.86
    longitude             (date_time) float64 16B 14.89 14.89
Attributes:
    description:       This is the ultimate first version of meteorologic dat...
    VERSION:           0.0.0
    station_priority:  ['0-203-0-20407036001', '0-203-0-11601', '0-20000-0-11...
    notes:             Preliminary CHMI-to-ParFlow/CLM forcing conversion. DS...
    time_step:         3600000000000
    time_interval:     189385200000000000
    __structure__:     ATTRS:\n  description: This is the ultimate first vers...
sqrt func call
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
[UKF] R shape: (5, 5); measurement shape: (5,)
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362984]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362845]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591344 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07592245 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07833828 0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832104 0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08226636 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08562182 0.08226021 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08560899 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362078]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362217]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07590979 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07590081 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07830177 0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07831896 0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08222614 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08556038 0.08223228 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08557315 0.08224621 0.07832    0.07591161 0.07362531]
measurement function output size =   [0.08559102 0.08224621 0.07832    0.07591161 0.07362531]
[UKF] Covariance sum: 59.375758411855436
[UKF] State estimate: [ -8.66806931 -95.62627191 -91.42387894 -84.8679773  -81.27035513
 -78.56673593 -74.28102175 -69.4476354  -62.54465125 -61.42386196
 -48.22571432 -52.29854729 -38.6887671  -19.49995396 -27.31768354
  -2.61058112]
[UKF] Step 4: 2025-03-06T03:00:00.000000000 → 2025-03-06T04:00:00.000000000
meteo_ds  <xarray.Dataset> Size: 420B
Dimensions:               (date_time: 2)
Coordinates:
  * date_time             (date_time) datetime64[ns] 16B 2025-03-06T03:00:00 ...
    site_id               int32 4B 1
Data variables: (12/18)
    APCP                  (date_time) float64 16B 0.0 0.0
    D10_source_station    (date_time) StringDType() 32B '0-203-0-20407036001'...
    DLWR                  (date_time) float64 16B 165.2 165.7
    DSWR                  (date_time) float64 16B 0.0 0.0
    F_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    H_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    ...                    ...
    Temp                  (date_time) float64 16B 276.8 276.6
    UGRD                  (date_time) float64 16B 0.05585 0.2239
    VGRD                  (date_time) float64 16B 3.2 3.202
    elevation             (date_time) float64 16B 303.0 303.0
    latitude              (date_time) float64 16B 50.86 50.86
    longitude             (date_time) float64 16B 14.89 14.89
Attributes:
    description:       This is the ultimate first version of meteorologic dat...
    VERSION:           0.0.0
    station_priority:  ['0-203-0-20407036001', '0-203-0-11601', '0-20000-0-11...
    notes:             Preliminary CHMI-to-ParFlow/CLM forcing conversion. DS...
    time_step:         3600000000000
    time_interval:     189385200000000000
    __structure__:     ATTRS:\n  description: This is the ultimate first vers...
sqrt func call
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
[UKF] R shape: (5, 5); measurement shape: (5,)
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07369302]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07369158]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606248 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07607192 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07911328 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909341 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08622789 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09349637 0.08623753 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09344696 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368372]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368515]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07605877 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07604935 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07907149 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.0790913  0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08617946 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09334777 0.08616998 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09339673 0.08620362 0.07909235 0.07606062 0.07368836]
measurement function output size =   [0.09342177 0.08620362 0.07909235 0.07606062 0.07368836]
[UKF] Covariance sum: 60.46401691573439
[UKF] State estimate: [ -7.72037387 -95.51983232 -91.423878   -84.05544051 -80.70920184
 -78.56673507 -74.28102102 -69.25617587 -61.37812445 -61.42384219
 -44.37784917 -52.10426795 -34.07582395   8.68868751 -19.96561049
  -2.61058113]
[UKF] Step 5: 2025-03-06T04:00:00.000000000 → 2025-03-06T05:00:00.000000000
meteo_ds  <xarray.Dataset> Size: 420B
Dimensions:               (date_time: 2)
Coordinates:
  * date_time             (date_time) datetime64[ns] 16B 2025-03-06T04:00:00 ...
    site_id               int32 4B 1
Data variables: (12/18)
    APCP                  (date_time) float64 16B 0.0 0.0
    D10_source_station    (date_time) StringDType() 32B '0-203-0-20407036001'...
    DLWR                  (date_time) float64 16B 165.7 166.0
    DSWR                  (date_time) float64 16B 0.0 0.0
    F_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    H_source_station      (date_time) StringDType() 32B '0-203-0-20407036001'...
    ...                    ...
    Temp                  (date_time) float64 16B 276.6 276.6
    UGRD                  (date_time) float64 16B 0.2239 0.4241
    VGRD                  (date_time) float64 16B 3.202 3.454
    elevation             (date_time) float64 16B 303.0 303.0
    latitude              (date_time) float64 16B 50.86 50.86
    longitude             (date_time) float64 16B 14.89 14.89
Attributes:
    description:       This is the ultimate first version of meteorologic dat...
    VERSION:           0.0.0
    station_priority:  ['0-203-0-20407036001', '0-203-0-11601', '0-20000-0-11...
    notes:             Preliminary CHMI-to-ParFlow/CLM forcing conversion. DS...
    time_step:         3600000000000
    time_interval:     189385200000000000
    __structure__:     ATTRS:\n  description: This is the ultimate first vers...
sqrt func call
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
dt:  1 number of model iterations:  40
process PID: 62 thread: 130034580469440
```


## AGENT log
- `2026-06-03`: Added main-process composed/Kalman logging for the `runs/composed_1d_only` infinite-loop investigation: `calculation.log` in workdir from `hlavo.main`, INFO progress for key 1D-3D and Kalman time iterations, DEBUG array diagnostics, no worker-side global file logging setup, main-process logging for re-raised worker failures, and assertions for non-advancing 1D/3D target times.
- `2026-06-03`: Documented `ToyProblem` and `ToyProblem.run()` contracts in `hlavo/soil_parflow/parflow_model.py`, including config keys, fixed runtime files, CLM forcing requirements, run directory behavior, and Kalman-facing outputs.
- `2026-06-03`: Resolved `AGENTS.md` review answers from QaR: staging user edits is intentional but commits are forbidden unless explicitly requested; `AGENT` wording is the unified marker; `AGENT log` is for completed records while QaR is for unresolved user-facing inconsistencies; coding rules not present in `python_coding.md` were restored in `AGENTS.md`.
- `2026-06-03`: Reviewed `AGENTS.md` Workflow and in-source communication split. Removed duplicated staging, `AGENT` handling, large-change planning, and Python coding-rule references from `AGENTS.md`. No unresolved Workflow inconsistency remains after the cleanup.

## AGENT Questions And Remarks
