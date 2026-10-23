# Question: `zarr_fuse.open_store()` writes on every open

Status: decided 2026-10-16 (option D below); the read-only open in zarr_fuse (option A) stays
a possible later improvement.

## Problem

`zarr_fuse.open_store()` is not a read-only operation. Opening a store rewrites the group
metadata of every node whose stored schema is empty, which includes container nodes such as
`Uhelna`. There is no way to open a store read-only: the `MODE` option is read, but it is not
passed to the store.

Consequences for HLAVO:
- Two 1D workers (separate Dask worker processes) that open the same input store at the same
  time race on these writes. One of them fails with
  `zarr.errors.ContainsGroupError: A group exists in store ... at path 'Uhelna'`.
- Every 1D worker writes into the shared S3 input stores (`s3://hlavo-release/...`) just by
  reading its inputs.

## Evidence

Run `runs/composed_3d_writer` (branch `codex/m2-zarr-coupled-output`, 2 sites, local input
stores created completely beforehand by `prepare_inputs.py`), `runs/composed_3d_writer/run.log`:

```
Model1D.from_config -> Model1DData.from_config -> load_meteo_data -> zf.open_store
  -> Node("") -> _make_consistent -> Node("Uhelna") -> _make_consistent
  -> _update_schema_ds -> _init_empty_grup -> zarr.open_group(mode="a")
  -> ContainsGroupError ... at path 'Uhelna'
```
The zarr warnings about `.partial` objects seen in earlier runs point to the same concurrent writes.

Code (installed zarr_fuse `e336716`, `zarr_fuse/zarr_storage.py`; the same code is in zarr_fuse
`main`, `a2a23c5`):
- `open_store()`: `mode = options.get('MODE', 'a')` is passed only to `Node`, not to the store.
- `_zarr_store_open()`: always `zarr.storage.LocalStore(path)` / `FsspecStore(fs, path=...)`,
  i.e. writable.
- `Node.__init__()`: switches to `mode = 'r'` only if `store.read_only`, which never happens.
- `Node._update_schema_ds()`: if the stored schema is missing or `is_empty()`, calls
  `_init_empty_grup()`, which writes the group, regardless of `mode`.

Callers in HLAVO that only read: `load_measurments_data()` and `load_meteo_data()` in
`hlavo/ingress/moist_profile/load_zarr_data.py` (every 1D worker), `_load_wells_dataset()` in
`hlavo/composed/prediction_writer.py`.

## Options

### A. Read-only open in zarr_fuse (proper fix)
zarr_fuse:
```python
def _zarr_store_open(store_options):
    read_only = store_options.get('MODE', 'a') == 'r'
    ...
    return zarr.storage.FsspecStore(fs, path=clean_path, read_only=read_only)
    ...
    return zarr.storage.LocalStore(path, read_only=read_only)

# Node._update_schema_ds(): with self.mode == 'r' never initialize groups; return the stored
# schema, fail if the node does not exist.
```
HLAVO: the read-only callers above pass `MODE='r'` to `zf.open_store()`.
- Pros: removes the race and the writes to the shared S3 stores; small change.
- Cons: change in GeoMop/zarr_fuse (review, release, reinstall in the docker venv) before the
  run works.

### B. Serialize the input loading in HLAVO (workaround)
```python
from distributed import Lock
with Lock("hlavo-input-open"):  # zarr_fuse.open_store writes on open, see this note
    profiles = select(load_measurments_data(...))
    surface = select(load_meteo_data(...))
```
in `Model1DData.from_config()`.
- Pros: small, HLAVO only, works now.
- Cons: still writes on every open (also to S3); needs a distributed client, so `Model1D` used
  outside a Dask cluster (e.g. directly in a test) needs special handling.

### C. One site in the run for now
`site_ids: [1]` in `runs/composed_3d_writer/config.yaml`.
- Pros: no code change, the run works today.
- Cons: does not exercise concurrent 1D workers; the problem stays for real runs.

### D. All ZARR access in the main process (chosen)
The main process, which runs the 3D model and the prediction writer, reads the inputs of all
1D sites before the 1D workers start (`Model1D.load_inputs` -> `Model1DData.load_sites`: each
input store opened once, data selected per site and loaded into memory) and passes each worker
the data of its site (`model1d_worker_entry(..., inputs, ...)`). The workers open no store.
- Pros: no concurrent opens, no writes from the workers into the shared S3 input stores, works
  with the current zarr_fuse; matches the design that only the 3D side writes ZARR.
- Cons: the main process still writes group metadata once per store when opening it (zarr_fuse);
  the site data travel to the workers through Dask (fine for weeks to months of data; for long
  runs consider `client.scatter`).

## Decision

2026-10-16 (Otto Severýn with Jan Březina): option D. Implemented in
`hlavo/kalman/model_1d.py` (`load_inputs` of the 1D model classes), `hlavo/composed/worker_1d.py`
and `hlavo/composed/model_composed.py`.
