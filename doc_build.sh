#!/usr/bin/env bash
set -euo pipefail

python=".venv/bin/python"
sphinx_build=".venv/bin/sphinx-build"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  $python -m pip install -r doc/requirements.txt
fi

$sphinx_build -b html doc doc/_build/html
ls -l doc/_build/html/index.html