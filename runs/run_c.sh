#!/usr/bin/env bash

set -euo pipefail

# Use this script to run hlavo under CONDA environment.

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

HLAVO_CLI="$SCRIPT_ROOT/../dev/hlavo"
HLAVO_MAIN="$SCRIPT_ROOT/../hlavo/main.py"
exec "$HLAVO_CLI" -c run python3 "$HLAVO_MAIN" "$@"
