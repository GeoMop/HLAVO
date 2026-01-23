#!/bin/bash
#PBS -N gprMax
#PBS -l select=1:ncpus=20:mem=90gb
#PBS -j oe

module load mambaforge


## Installation

## Clone gprMax git:
# git clone https://github.com/gprMax/gprMax.git


cd $HOME/workspace/gprMax


## Prepare environment
# mamba env create -f conda_env.yml

## or (older slower interface)
# conda env create -f conda_env.yml


## Activate environment
conda activate gprMax

## Build and Install
#python setup.py build
#python setup.py install

# or editable installation

#python -m pip install -e . 

## Run gprMax and processing

cd HLAVO/small_scale/
bash run.sh
