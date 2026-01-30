#!/bin/bash

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


# LaTeX sources
"${SCRIPT_ROOT}/../dev/hlavo" -c run sphinx-build -b latex "${SCRIPT_ROOT}" "${SCRIPT_ROOT}/_build/latex"
