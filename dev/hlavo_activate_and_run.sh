#!/usr/bin/env bash
set -xeuo pipefail

# This wrapper is injected by run_cmd before the user command.
# It keeps base_run_* as raw environment entry points while still ensuring
# every normal command executes from the overlay venv passed in $1.

[[ $# -ge 2 ]] || {
  echo "ERROR: Usage: $0 <venv_dir> <command> [args...]" >&2
  exit 1
}

venv_dir="$1"
shift

echo "Activating VENV: $venv_dir" 
source "$venv_dir/bin/activate"
exec "$@"
