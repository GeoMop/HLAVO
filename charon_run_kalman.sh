#!/bin/bash

set -x
pbs_script=`pwd`/run_kalman.pbs

work_dir=$1

HLAVODIR="/storage/liberec3-tul/home/martin_spetlik/HLAVO"

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=20:cgroups=cpuacct:scratch_local=32gb:mem=32Gb
#PBS -l walltime=48:00:00
#PBS -q charon
#PBS -N kalman_soil_model_transform_param_n_exp_1
#PBS -j oe


cd ${work_dir}

singularity exec -B ${HLAVODIR} -B ${work_dir} /storage/liberec3-tul/home/martin_spetlik/kalman_parflow.sif python3 "${HLAVODIR}/soil_model/kalman.py" ${work_dir} config_0_experiments.yaml
EOF

qsub $pbs_script

