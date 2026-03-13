#!/bin/bash
#PBS -N dask_map_demo
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=1gb
#PBS -l place=scatter
#PBS -l walltime=2:00:00
#PBS -j oe

set -euo pipefail
set -x

############################################
# CONFIGURATION
############################################

work_dir="/storage/liberec3-tul/home/martin_spetlik/HLAVO_distr/runs/deep_model"
HLAVODIR="/storage/liberec3-tul/home/martin_spetlik/HLAVO_distr"
CONTAINER="/storage/liberec3-tul/home/martin_spetlik/hlavo_0_1_0.sif"

ENV_PY="/home/hlavo/miniconda3/envs/hlavo/bin/python"

SCHED_FILE="$PBS_O_WORKDIR/scheduler.json"

cd "$PBS_O_WORKDIR"

echo "Job started at: $(date)"
echo "Host: $(hostname -f)"
echo "CPUs allocated: ${PBS_NCPUS:-4}"
echo "SCRATCHDIR: ${SCRATCHDIR:-<not-set>}"

############################################
# CLEANUP HANDLER
############################################

cleanup() {
  echo "Stopping scheduler..."
  [[ -n "${SCHED_PID:-}" ]] && kill "$SCHED_PID" 2>/dev/null || true
  wait || true
}
trap cleanup EXIT

############################################
# START DASK SCHEDULER
############################################

: > "$SCHED_FILE"

echo "Starting Dask scheduler..."

singularity exec \
  -B "$HLAVODIR" \
  -B "$work_dir" \
  -B "$SCRATCHDIR" \
  "$CONTAINER" \
  "$ENV_PY" -m distributed.cli.dask_scheduler \
    --scheduler-file "$SCHED_FILE" \
    --host "$(hostname -f)" \
    --protocol tcp \
    --port 0 \
    --dashboard-address ":0" \
    > scheduler.log 2>&1 &

SCHED_PID=$!

echo -n "Waiting for scheduler"
for i in {1..40}; do
  if [[ -s "$SCHED_FILE" ]] && grep -q '"address"' "$SCHED_FILE"; then
    echo " - ready"
    break
  fi
  sleep 0.5
  echo -n "."
done

if ! grep -q '"address"' "$SCHED_FILE"; then
  echo "Scheduler failed to start"
  tail -n 200 scheduler.log || true
  exit 1
fi

############################################
# START DASK WORKERS (SINGLE NODE)
############################################

echo "Starting Dask workers..."

NWORKERS=${PBS_NCPUS:-4}

for i in $(seq 1 "$NWORKERS"); do
  singularity exec \
    -B "$HLAVODIR" \
    -B "$work_dir" \
    -B "$SCRATCHDIR" \
    "$CONTAINER" \
    "$ENV_PY" -m distributed.cli.dask_worker \
      --scheduler-file "$SCHED_FILE" \
      --nthreads 1 \
      --memory-limit 0 \
      --local-directory "$SCRATCHDIR" \
    > worker_${i}.log 2>&1 &
done

sleep 5

############################################
# RUN CLIENT
############################################

echo "Running composed model..."

singularity exec $CONTAINER $ENV_PY -m pip install joblib

singularity exec \
  -B "$HLAVODIR" \
  -B "$work_dir" \
  -B "$SCRATCHDIR" \
  "$CONTAINER" \
  env PYTHONPATH="$HLAVODIR" \
  "$ENV_PY" -m hlavo.composed.composed_model_mock \
    "${work_dir}/deep_model_results" \
    "${work_dir}/deep_model_config.yaml"

echo "Job finished at: $(date)"