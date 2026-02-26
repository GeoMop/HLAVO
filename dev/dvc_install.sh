#!/bin/bash

# Exit on any error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define variables
VENV_DIR="${SCRIPT_DIR}/dvc"
BIN_DIR="${HOME}/bin"
DVC_SYMLINK="${BIN_DIR}/dvc"

# Step 1: Create a virtual environment
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Step 2: Activate the virtual environment and install DVC
echo "Activating virtual environment and installing DVC..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install fsspec gdrivefs 
pip install dvc[s3,gs,gdrive]


deactivate

# Step 3: Create a symlink for the DVC executable
DVC_EXECUTABLE="$VENV_DIR/bin/dvc"
if [ -L "$DVC_SYMLINK" ]; then
    echo "Removing existing symlink at $DVC_SYMLINK..."
    rm "$DVC_SYMLINK"
fi

echo "Creating symlink for DVC at $DVC_SYMLINK..."
ln -s "$DVC_EXECUTABLE" "$DVC_SYMLINK"

# Step 4: Verify the installation
echo "Verifying DVC installation..."
if command -v dvc &>/dev/null; then
    echo "DVC successfully installed. Version:"
    dvc --version
    pip show dvc
    pip show dvc-gdrive
else
    echo "DVC installation failed. Please check for errors."
    exit 1
fi

echo "Installation complete. You can now use 'dvc' globally."
