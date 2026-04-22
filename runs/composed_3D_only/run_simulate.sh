#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
RUNS_DIR="$( dirname "$SCRIPT_DIR" )"
CONFIG_PATH="$SCRIPT_DIR/config.yaml"

bash "$RUNS_DIR/run.sh" simulate "$CONFIG_PATH" -w "$SCRIPT_DIR"
