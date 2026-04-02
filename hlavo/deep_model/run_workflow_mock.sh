#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH="${1:-config/base/mock_cube.yaml}"

echo ">>> python build_mock_cube.py --config ${CONFIG_PATH}"
python build_mock_cube.py --config "${CONFIG_PATH}"

echo ">>> python run_model.py --config ${CONFIG_PATH}"
python run_model.py --config "${CONFIG_PATH}"

echo ">>> python create_paraview.py --config ${CONFIG_PATH}"
python create_paraview.py --config "${CONFIG_PATH}"

echo ">>> python create_plots.py --config ${CONFIG_PATH}"
python create_plots.py --config "${CONFIG_PATH}"

echo "Mock workflow finished for config: ${CONFIG_PATH}"
