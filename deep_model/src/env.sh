#!/bin/bash
#
# Usage:
# ./mamba_env.sh [--force] 
#
# Install conda environment according to the 'conda-requirements.yml'
# The YAML config provides: name, source channel (conda-forge), list of installed packages

set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Environment for running fittings Jupyter notebook.
# Uses Conda environment
CONDA_PATH="/opt/miniconda"

# Function to install Miniconda
install_miniconda() {
    # Update package list and install prerequisites
    sudo apt update
    sudo apt install -y wget bzip2

    # Define Miniconda installer URL and download.
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    INSTALLER=/tmp/${MINICONDA_URL##*/}
    wget $MINICONDA_URL -O $INSTALLER
    
    # Run the installer silently (-b) and specify custom installation path (-p)
    sudo bash $INSTALLER -b -p $CONDA_PATH
    
    # Ensure the installation directory is writable by the current user
    sudo chown -R $USER:$USER $CONDA_PATH
    
    # Initialize conda for bash shell
    $CONDA_PATH/bin/conda init bash
        
    # Refresh the shell
    source ~/.bashrc
    
    echo "Miniconda installed at $CONDA_PATH."
}

# Main script execution
# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Skipping Miniconda installation."
else
    install_miniconda
fi



# Ensure conda commands work in this script
source $CONDA_PATH/etc/profile.d/conda.sh

# Install mamba in the base environment if not already present
if ! command -v mamba &> /dev/null; then
    echo "Installing mamba in the base environment..."
    conda install -y mamba -n base -c conda-forge
fi

# Extract the environment name from the YAML file
env_yaml="$SCRIPTPATH/conda-requirements.yml"
env_name=$(grep '^name:' "$env_yaml" | awk '{print $2}')

# Use mamba to remove, create, or update the environment
if [ "$1" == "create" ]; then
    mamba env remove -n $env_name
    mamba env create -y --file "$env_yaml" 
    exit 0
elif [ "$1" == "update" ]; then
    mamba env update -y --file "$env_yaml"
    exit 0
elif  [ "$1" == "list" ]; then
    conda info --envs
elif [ "$1" == "run" ]; then
    conda activate $env_name
    shift
    $@
elif [ "$1" == "conda" ]; then
    shift 
    conda $@
fi



# Install the IPython kernel for Jupyter
#ipython kernel install --name "$env_name" --user

