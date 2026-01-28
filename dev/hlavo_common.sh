#!/usr/bin/env bash
set -euo pipefail

COMMON_ROOT="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="${REPO_ROOT:-$COMMON_ROOT}"
IMAGE_NAME="hlavo"
DOCKERFILE="$REPO_ROOT/hlavo_dockerfile"
HLAVO_MODE="${HLAVO_MODE:-docker}"
ENV_YAML="${ENV_YAML:-$REPO_ROOT/conda-requirements.yml}"
REQ_TXT="${REQ_TXT:-$REPO_ROOT/requirements.txt}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/..}"
PYPROJECT="${PYPROJECT:-$PROJECT_ROOT/pyproject.toml}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_BIN="${CONDA_BIN:-$CONDA_BASE/bin/conda}"
MAMBA_BIN="${MAMBA_BIN:-$CONDA_BASE/bin/mamba}"

die() { echo "ERROR: $*" >&2; exit 1; }

parse_version_from_pyproject() {
  awk -F'=' '
    /^[[:space:]]*\[project\][[:space:]]*$/ { in_project=1; next }
    in_project && /^[[:space:]]*\[/ { in_project=0 }
    in_project && /^[[:space:]]*version[[:space:]]*=/ {
      v=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      gsub(/^["'\'']|["'\'']$/, "", v)
      print v
      exit
    }' "$PYPROJECT"
}

parse_env_name_from_yaml() {
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

ENV_NAME="${ENV_NAME:-$(parse_env_name_from_yaml)}"

set_image_vars() {
  local version
  if [[ -n "${IMAGE_REF:-}" ]]; then
    return 0
  fi
  version=""
  if [[ -f "$PYPROJECT" ]]; then
    version="$(parse_version_from_pyproject)"
  fi
  IMAGE_TAG="${IMAGE_TAG:-${version:-latest}}"
  IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    CONDA_BASE="$("$CONDA_BIN" info --base)"
    MAMBA_BIN="$CONDA_BASE/bin/mamba"
    return 0
  fi

  if [[ -x "$CONDA_BIN" ]]; then
    return 0
  fi

  command -v wget >/dev/null 2>&1 || die "wget not found. Install it and re-run."
  command -v bzip2 >/dev/null 2>&1 || die "bzip2 not found. Install it and re-run."

  local os arch base url installer
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)  base="Linux" ;;
    Darwin) base="MacOSX" ;;
    *) die "Unsupported OS: $os" ;;
  esac

  case "$arch" in
    x86_64|amd64)   arch="x86_64" ;;
    aarch64|arm64)  arch="arm64"; [[ "$base" == "Linux" ]] && arch="aarch64" ;;
    *) die "Unsupported architecture: $arch (OS=$os)" ;;
  esac

  url="https://repo.anaconda.com/miniconda/Miniconda3-latest-${base}-${arch}.sh"
  installer="/tmp/${url##*/}"

  wget -q "$url" -O "$installer"
  bash "$installer" -b -p "$CONDA_BASE"
  [[ -x "$CONDA_BIN" ]] || die "Miniconda install failed (conda not found at $CONDA_BIN)."
}

ensure_mamba() {
  if [[ -x "$MAMBA_BIN" ]]; then
    return 0
  fi

  "$CONDA_BIN" config --remove channels defaults || true
  "$CONDA_BIN" config --add channels conda-forge
  "$CONDA_BIN" config --set channel_priority strict

  "$CONDA_BIN" install -y -n base -c conda-forge --override-channels mamba
  [[ -x "$MAMBA_BIN" ]] || die "mamba install failed (expected $MAMBA_BIN)."
}

conda_env_exists() {
  "$MAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

base_build_conda() {
  ensure_conda
  ensure_mamba
  if [[ "${1:-}" == "-f" ]]; then
    if conda_env_exists; then
      "$MAMBA_BIN" env remove -y -n "$ENV_NAME"
    fi
  fi
  "$MAMBA_BIN" env create -y --file "$ENV_YAML"
}

base_build_docker() {
  set_image_vars
  if [[ "${1:-}" == "-f" ]]; then
    if docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
      docker rmi -f "$IMAGE_REF"
    fi
  fi
  docker build -f "$DOCKERFILE" -t "$IMAGE_REF" "$REPO_ROOT"
}

base_run_conda() {
  ensure_conda
  "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

base_run_docker() {
  local img_workspace img_conda_path tty_arg

  set_image_vars
  command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
  img_workspace="${HLAVO_WORKSPACE:-/home/hlavo/workspace}"
  img_conda_path="/home/hlavo/miniconda3/bin/conda"
  tty_arg="${TERM:-}"
  if [[ "$tty_arg" != -* ]]; then
    tty_arg=""
  fi

  docker run --rm \
    ${tty_arg} \
    -e HLAVO_UID="$(id -u)" \
    -e HLAVO_GID="$(id -g)" \
    -v "$REPO_ROOT:$img_workspace" \
    -w "$img_workspace" \
    "$IMAGE_REF" \
    "$img_conda_path" run -n "$ENV_NAME" "$@"
}

env_build() {
  base_build "$@"
  venv_ensure "$@"
  venv_overlay
}

run_cmd() {
  venv_ensure
  [[ $# -ge 1 ]] || die "Missing command."
  cmd="$(printf '%q ' "$@")"
  base_run bash -lc "source \"$VENV_DIR/bin/activate\"; exec $cmd"
}

case "$HLAVO_MODE" in
  conda)
    VENV_DIR="$REPO_ROOT/venv-conda"
    base_run() { base_run_conda "$@"; }
    base_build() { base_build_conda "$@"; }
    ;;
  docker)
    VENV_DIR="${HLAVO_WORKSPACE:-/home/hlavo/workspace}/venv-docker"
    base_run() { base_run_docker "$@"; }
    base_build() { base_build_docker "$@"; }
    ;;
  *)
    die "Unknown HLAVO_MODE: $HLAVO_MODE"
    ;;
esac


# following down to venv_* functions probably not neccessary

ensure_conda_shell() {
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}


# ----

venv_ensure() {
  if [[ "${1:-}" == "-f" ]]; then
    rm -rf "$VENV_DIR"
  elif [[ -x "$VENV_DIR/bin/python" ]]; then
    return 0
  fi
  base_run python -m venv --system-site-packages "$VENV_DIR"
  base_run "$VENV_DIR/bin/python" -m pip install --upgrade pip
}

venv_overlay() {
  venv_ensure
  if [[ -f "$REQ_TXT" ]]; then
    base_run "$VENV_DIR/bin/python" -m pip install -r "$REQ_TXT"
  fi

  if [[ -f "$PYPROJECT" ]]; then
    base_run "$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT"
  else
    echo "Note: $PYPROJECT not found — skipping editable install."
  fi
}

venv_rebuild() {
  rm -rf "$VENV_DIR"
  venv_overlay
}
