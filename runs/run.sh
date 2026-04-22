#!/usr/bin/env bash

set -euo pipefail

# Use to run hlavo under docker environment.

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
HLAVO_CLI="$SCRIPT_ROOT/../dev/hlavo"

exec "$HLAVO_CLI" run hlavo "$@"
