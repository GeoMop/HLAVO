#!/bin/bash

SCRIPT_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


${SCRIPT_ROOT}/../dev/hlavo -c run sphinx-build -b html "${SCRIPT_ROOT}" "${SCRIPT_ROOT}/_build/html"
