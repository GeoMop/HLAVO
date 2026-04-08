#!/usr/bin/env bash

set -euo pipefail

# Use this script to run under CODEX (or within a HLAVO environment).

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

HLAVO_MAIN="$SCRIPT_ROOT/../hlavo/main.py"
exec python3 "$HLAVO_MAIN" "$@"
