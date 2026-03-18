#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <config_path> [workspace_override]" >&2
  exit 2
fi

CONFIG_PATH="$1"
WORKSPACE_OVERRIDE="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

run_cmd() {
  echo ">>> $*"
  "$@"
}

if [[ -n "${WORKSPACE_OVERRIDE}" ]]; then
  run_cmd python build_modflow_grid.py --config "${CONFIG_PATH}"
  run_cmd python add_material_parameters.py --config "${CONFIG_PATH}"
  run_cmd python run_model.py --config "${CONFIG_PATH}" --workspace "${WORKSPACE_OVERRIDE}"
  run_cmd python visualize_results.py --config "${CONFIG_PATH}" --workspace "${WORKSPACE_OVERRIDE}"
else
  run_cmd python build_modflow_grid.py --config "${CONFIG_PATH}"
  run_cmd python add_material_parameters.py --config "${CONFIG_PATH}"
  run_cmd python run_model.py --config "${CONFIG_PATH}"
  run_cmd python visualize_results.py --config "${CONFIG_PATH}"
fi

echo "Workflow finished for config: ${CONFIG_PATH}"
