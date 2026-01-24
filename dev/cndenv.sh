#!/usr/bin/env bash
#
# cndenv.sh — single entrypoint conda based environment
#
# Behavior:
#   - If conda-requirements.yml exists in the repo -> use Miniconda + mamba env
#   - If not -> use local Python venv at ./venv (no conda required)
#
# Usage:
#   ./env.sh help
#   ./env.sh update
#   ./env.sh rebuild
#   ./env.sh shell
#   ./env.sh list
#   ./env.sh run -- <command> [args...]
#   ./env.sh conda -- <conda args...>     (conda-mode only)
#   ./env.sh mamba -- <mamba args...>     (conda-mode only; venv-mode gives a friendly error)
#
# Notes:
#   - No .bashrc modifications
#   - No sudo usage (user-space Miniconda install)
#   - Works from any working directory (uses script location as rfiepo root)
#
# Optional:
#   CONDA_BASE=/custom/path   # override Miniconda install location (default: $HOME/miniconda3)


set -euo pipefail
set -x

REPO_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

ENV_YAML="$REPO_ROOT/conda-requirements.yml"
REQ_TXT="$REPO_ROOT/requirements.txt"
VENV_DIR="$REPO_ROOT/venv"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_BASE/bin/conda"
MAMBA_BIN="$CONDA_BASE/bin/mamba"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  ./env.sh help
  ./env.sh update
  ./env.sh rebuild
  ./env.sh shell
  ./env.sh list
  ./env.sh run -- <command> [args...]
  ./env.sh conda -- <conda args...>     (conda-mode only)
  ./env.sh mamba -- <mamba args...>     (conda-mode only)

Mode selection:
  - If conda-requirements.yml exists in the repo: conda/mamba mode
  - Otherwise: python venv mode at ./venv
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# --- Conda/Mamba bootstrap ----------------------------------------------------

install_miniconda_user() {
  need_cmd wget
  need_cmd bzip2
  need_cmd uname
  need_cmd bash

  local os arch variant base url installer

  os="$(uname -s)"
  arch="$(uname -m)"

  # Detect WSL (still uses Linux installers)
  if [[ "$os" == "Linux" ]] && { [[ -n "${WSL_INTEROP:-}" ]] || grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; }; then
    variant="WSL"
  else
    variant="$os"
  fi

  # Map OS to Miniconda naming
  case "$os" in
    Linux)  base="Linux" ;;
    Darwin) base="MacOSX" ;;
    *)
      echo "Unsupported OS: $os" >&2
      return 1
      ;;
  esac

  # Map architecture to Miniconda naming
  case "$arch" in
    x86_64|amd64)   arch="x86_64" ;;
    aarch64|arm64)  arch="arm64"
      # Miniconda uses aarch64 for Linux, arm64 for macOS
      [[ "$base" == "Linux" ]] && arch="aarch64"
      ;;
    *)
      echo "Unsupported architecture: $arch (OS=$os)" >&2
      return 1
      ;;
  esac

  url="https://repo.anaconda.com/miniconda/Miniconda3-latest-${base}-${arch}.sh"
  installer="/tmp/${url##*/}"

  echo "Detected: ${variant} (${os}/${arch})"
  echo "Downloading: $url"

  wget -q "$url" -O "$installer"
  bash "$installer" -b -p "$CONDA_BASE"
}


ensure_conda() {
  if [[ -x "$CONDA_BIN" ]]; then
    return 0
  fi

  # Best-effort: ensure wget/bzip2 exist (no sudo). If missing, user must install.
  if ! command -v wget >/dev/null 2>&1; then
    die "wget not found. Install it (e.g., apt install wget) and re-run."
  fi
  if ! command -v bzip2 >/dev/null 2>&1; then
    die "bzip2 not found. Install it (e.g., apt install bzip2) and re-run."
  fi

  echo "Miniconda not found at $CONDA_BASE — installing (user-space)..."
  install_miniconda_user
  [[ -x "$CONDA_BIN" ]] || die "Miniconda install failed (conda not found at $CONDA_BIN)."
}

ensure_conda_shell() {
  # Enables `conda activate` in this non-interactive script without .bashrc.
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}

ensure_mamba() {
  # We use conda only for bootstrapping mamba into base. After that we use mamba.
  if [[ -x "$MAMBA_BIN" ]]; then
    return 0
  fi

  echo "mamba not found — installing into base (conda-forge)..."
  "$CONDA_BIN" install -y -n base -c conda-forge mamba
  [[ -x "$MAMBA_BIN" ]] || die "mamba install failed (expected $MAMBA_BIN)."
}

parse_env_name_from_yaml() {
  # Robust enough for common cases: name: foo / name: "foo" / name: 'foo'
  local name
  name="$(
    awk -F': *' '
      /^[[:space:]]*name[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        gsub(/^["'\'']|["'\'']$/, "", v)
        print v
        exit
      }' "$ENV_YAML"
  )"
  [[ -n "${name:-}" ]] || die "Could not parse environment name from $ENV_YAML (missing/invalid 'name: ...')."
  echo "$name"
}

conda_env_exists() {
  local env_name="$1"
  # mamba env list prints names in first column; match whole word.
  "$MAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$env_name"
}

# --- venv backend (mamba "mock") ---------------------------------------------

venv_ensure() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    return 0
  fi
  need_cmd python3
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
}

venv_install_requirements() {
  if [[ -f "$REQ_TXT" ]]; then
    conda activate "$env_name"
    "$VENV_DIR/bin/python" -m pip install -r "$REQ_TXT"
  else
    echo "Note: $REQ_TXT not found — skipping pip installs."
  fi
}

# venv backend implements "mamba-like" operations via functions to keep one script logic.

backend_update() {
  if [[ -f "$ENV_YAML" ]]; then
    echo "Mode: conda/mamba (found conda-requirements.yml)"
    ensure_conda
    ensure_mamba
    ensure_conda_shell

    local env_name
    env_name="$(parse_env_name_from_yaml)"

    if conda_env_exists "$env_name"; then
      "$MAMBA_BIN" env update -y --file "$ENV_YAML" --prune
    else
      "$MAMBA_BIN" env create -y --file "$ENV_YAML"
    fi

    # Optional pip layer
    if [[ -f "$REQ_TXT" ]]; then
      conda activate "$env_name"
      python -m pip install -r "$REQ_TXT"
    else
      echo "Note: $REQ_TXT not found — skipping pip installs."
    fi
  else
    echo "Mode: python venv (no conda-requirements.yml)"
    venv_ensure
    venv_install_requirements
  fi
}

backend_rebuild() {
  if [[ -f "$ENV_YAML" ]]; then
    echo "Mode: conda/mamba (found conda-requirements.yml)"
    ensure_conda
    ensure_mamba
    ensure_conda_shell

    local env_name
    env_name="$(parse_env_name_from_yaml)"

    if conda_env_exists "$env_name"; then
      "$MAMBA_BIN" env remove -y -n "$env_name"
    fi
    "$MAMBA_BIN" env create -y --file "$ENV_YAML"

    if [[ -f "$REQ_TXT" ]]; then
      conda activate "$env_name"
      python -m pip install -r "$REQ_TXT"  
    else
      echo "Note: $REQ_TXT not found — skipping pip installs."
    fi
  else
    echo "Mode: python venv (no conda-requirements.yml)"
    rm -rf "$VENV_DIR"
    venv_ensure
    venv_install_requirements
  fi
}

backend_list() {
  if [[ -f "$ENV_YAML" ]]; then
    echo "Mode: conda/mamba (found conda-requirements.yml)"
    ensure_conda
    ensure_mamba
    "$MAMBA_BIN" env list
  else
    echo "Mode: python venv (no conda-requirements.yml)"
    if [[ -x "$VENV_DIR/bin/python" ]]; then
      "$VENV_DIR/bin/python" -V
      echo "venv path: $VENV_DIR"
    else
      echo "venv not created yet (expected at $VENV_DIR)"
    fi
  fi
}

backend_run() {
    ensure_conda
    ensure_mamba
    ensure_conda_shell
    env_name="$(parse_env_name_from_yaml)"

    (
      conda activate "$env_name"
      $VENV/bin/activate
      "$@"
    )
}

# 
# backend_shell() {
#     ensure_conda
#     ensure_mamba
#     ensure_conda_shell
#     local env_name
#     env_name="$(parse_env_name_from_yaml)"
#     # Spawn an interactive shell with env activated
#     # shellcheck disable=SC1090
#     ( conda activate "$env_name";  )
# }

backend_conda_passthrough() {
  [[ -f "$ENV_YAML" ]] || die "conda passthrough is only available in conda/mamba mode (requires conda-requirements.yml)."
  ensure_conda
  "$CONDA_BIN" "$@"
}

backend_mamba_passthrough() {
  [[ -f "$ENV_YAML" ]] || die "mamba passthrough is only available in conda/mamba mode (requires conda-requirements.yml)."
  ensure_conda
  ensure_mamba
  "$MAMBA_BIN" "$@"
}

# --- CLI ---------------------------------------------------------------------

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help) usage ;;
  update)         backend_update ;;
  rebuild|create) backend_rebuild ;;
  list)           backend_list ;;
  conda)
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -ge 1 ]] || die "conda passthrough needs args. Example: ./env.sh conda -- info"
    backend_conda_passthrough "$@"
    ;;
  mamba)
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -ge 1 ]] || die "mamba passthrough needs args. Example: ./env.sh mamba -- env list"
    backend_mamba_passthrough "$@"
    ;;

  # following two, installs venv first
  shell)          backend_run exec "${SHELL:-bash}" -i ;;
  run)
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -ge 1 ]] || die "run requires a command. Example: ./env.sh run -- python -V"
    backend_run "$@"
    ;;
  run)
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -ge 1 ]] || die "run requires a command. Example: ./env.sh run -- python -V"
    backend_run "$@"
    ;;
  *)
    usage
    die "Unknown command: $cmd"
    ;;
esac
