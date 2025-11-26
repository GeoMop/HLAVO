#!/bin/bash
#PBS -N dask_map_demo
#PBS -l select=2:ncpus=2:mem=4gb:scratch_local=1gb
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -j oe

# PBS variables in use:
# $SCRATCHDIR     - local scratch dir
# $PBS_NODEFILE   - list of nodes in multiplicity of ncpus  
# $PBS_O_WORKDIR  - where qsub is executed

set -euo pipefail
cd "$PBS_O_WORKDIR"

pwd
source venv/bin/activate
DASK=`which dask`


# --- Environment: ensure python + dask.distributed are available on ALL nodes ---
# If you use modules, uncomment and adapt:
# module load python/3.10
# If you have a shared venv, uncomment:
# source /path/to/venv/bin/activate

# Sanity: show nodes granted
echo "Nodes granted:"
sort "$PBS_NODEFILE" | uniq -c

SCHED_FILE="$PBS_O_WORKDIR/scheduler.json"
: > "$SCHED_FILE"  # ensure the file path exists


# Ensure we tear down everything on exit
cleanup() {
  echo "Stopping scheduler (if any)..."
  [[ -n "${SCHED_PID:-}" ]] && kill "$SCHED_PID" 2>/dev/null || true
  wait || true
}
trap cleanup EXIT

# 1) Start the Dask scheduler on this node (background)
echo "Starting dask-scheduler..."
$DASK scheduler \
  --scheduler-file "$SCHED_FILE" \
  --host "$(hostname -f)" \
  --protocol tcp \
  --port 0 \
  --dashboard-address ":0" \
  > scheduler.log 2>&1 &

SCHED_PID=$!


# Wait until the scheduler writes the scheduler file and is listening
echo -n "Waiting for scheduler to come up"
for i in {1..40}; do
  if [[ -s "$SCHED_FILE" ]] && grep -q '"address"' "$SCHED_FILE"; then
    echo " - ready."
    break
  fi
  sleep 0.5
  echo -n "."
done
if ! [[ -s "$SCHED_FILE" ]] || ! grep -q '"address"' "$SCHED_FILE"; then
  echo "Scheduler failed to come up; last lines of scheduler.log:"
  tail -n +1 -n 200 scheduler.log || true
  exit 1
fi

# 2) Start workers on each allocated slot using pbsdsh (1 worker per slot, 1 thread each)
#    If you prefer fewer, bump --nworkers and adjust --nthreads appropriately.
echo "Starting dask-workers via dsh..."
# Drop the first line (reserved for scheduler), keep multiplicity
tail -n +2 "$PBS_NODEFILE" > workerlist.txt
bash simple_dsh workerlist.txt $DASK worker \
  --scheduler-file "$SCHED_FILE" \
  --nworkers 1 \
  --nthreads 1 \
  --memory-limit 0 \
  --local-directory $SCRATCHDIR

# Note:
# - pbsdsh runs the command once per allocated slot across your nodes.
# - If pbsdsh is unavailable, see the SSH fallback below.

# Give workers a moment to connect
sleep 5

# 3) Run the Python client that submits the map workload
echo "Running run_map.py..."
python run_map.py

