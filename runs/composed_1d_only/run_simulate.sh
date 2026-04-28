#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
RUNS_DIR="$( dirname "$SCRIPT_DIR" )"
CONFIG_PATH="$SCRIPT_DIR/composed_config.yaml"
HLAVO_CLI="$SCRIPT_DIR/../../dev/hlavo"



bash "$RUNS_DIR/run.sh" simulate "$CONFIG_PATH" -w "$SCRIPT_DIR"


# Auxiliary execution of the worker only calculation in order to debug directly.
#exec "$HLAVO_CLI" run python -m hlavo.composed.worker_1d "$CONFIG_PATH" -w "$SCRIPT_DIR"
