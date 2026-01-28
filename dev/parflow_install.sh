#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARFLOW_REF="v3.14.1"
PARFLOW_URL="https://github.com/parflow/parflow/archive/refs/tags/${PARFLOW_REF}.tar.gz"
PARFLOW_PREFIX="${PARFLOW_PREFIX:-$SCRIPT_DIR/parflow_install}"

need_pkg=(
  ca-certificates
  build-essential
  gfortran
  cmake
  tcl
  tcl-dev
  openmpi-bin
  libopenmpi-dev
  wget
)

missing_pkgs=()
for pkg in "${need_pkg[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    missing_pkgs+=("$pkg")
  fi
done

if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  echo "Installing missing system packages: ${missing_pkgs[*]}"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${missing_pkgs[@]}"
  else
    apt-get update
    apt-get install -y "${missing_pkgs[@]}"
  fi
fi


workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "Downloading ParFlow source: $PARFLOW_URL"
wget -q -O "$workdir/parflow.tar.gz" "$PARFLOW_URL"
tar -xzf "$workdir/parflow.tar.gz" -C "$workdir"
src_dir="$(find "$workdir" -maxdepth 1 -type d -name "parflow-*" -print -quit)"
[[ -n "${src_dir:-}" ]] || { echo "Failed to locate ParFlow source dir."; exit 1; }

mkdir -p "$workdir/build" "$PARFLOW_PREFIX"
cd "$workdir/build"
cmake "$src_dir" \
  -DCMAKE_INSTALL_PREFIX="$PARFLOW_PREFIX" \
  -DPARFLOW_AMPS_LAYER=mpi1 \
  -DPARFLOW_HAVE_CLM=TRUE
make -j"$(nproc)"
make install
