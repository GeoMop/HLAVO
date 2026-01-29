#!/bin/bash
set -x 

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
EM_models_root="$SCRIPTPATH/../.."

conda activate gprMax

# Output geometry first
python -m gprMax --geometry-only main.in


OMP_NUM_THREADS=6
python -m gprMax main.in 

python "$EM_models_root/plot_antenna_params.py" main.out --rx-num 1 --rx-component Ez 
