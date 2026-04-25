#!/bin/bash
#
# Usage:
#       setup_env.sh [-d]
#
# Install the package together with its dependencies as well as the `dev` dependencies 
# (see [project.optional-dependencies])
#
# [-d] Force removal of the created venv.



set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Name of the virtual environment folder
VENV_DIR="$SCRIPT_DIR/venv"

# 
if [ "$1" == "-d" ];
then
    rm -rf "$VENV_DIR"
fi

which python3
python3 --version

# Check if virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
else
    # Create a virtual environment
    python3 -m venv $VENV_DIR
    echo "Virtual environment created."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip
pip install "git+https://github.com/GeoMop/zarr_fuse.git@feature/hlavo-integration"
pip install "git+https://github.com/GeoMop/zarr_fuse.git@feature/hlavo-integration#subdirectory=dashboard"
