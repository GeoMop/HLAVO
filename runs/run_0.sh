#!/usr/bin/env bash

set -euo pipefail

# Use this script to run under CODEX (or within a HLAVO environment).

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="$( dirname "$SCRIPT_ROOT" )"
cd "$REPO_ROOT"
exec python3 -m hlavo.main "$@"
