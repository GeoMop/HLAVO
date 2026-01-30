#!/usr/bin/env bash
set -xeuo pipefail

COMMON_ROOT="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
REPO_ROOT="${REPO_ROOT:-$COMMON_ROOT/..}"
IMAGE_NAME="flow123d/hlavo"
HLAVO_MODE="${HLAVO_MODE:-docker}"
ENV_YAML="$REPO_ROOT/dev/conda-requirements.yml"

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
    }' "$1"
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
  # Can not run within docker.
  local version
  if [[ -n "${IMAGE_REF:-}" ]]; then
    return 0
  fi
  PYPROJECT="$REPO_ROOT/pyproject.toml"
  if [[ -f "$PYPROJECT" ]]; then
    version="$(parse_version_from_pyproject $PYPROJECT)"
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
  
  conda_env_exists || "$MAMBA_BIN" env create -y --file "$ENV_YAML"
}

base_build_docker() {
  set_image_vars
  if [[ "${1:-}" == "-f" ]]; then
    if docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
      docker rmi -f "$IMAGE_REF"
    fi
  fi
  docker build -f "$REPO_ROOT/dev/hlavo_dockerfile" -t "$IMAGE_REF" "$REPO_ROOT/dev"
}

base_run_conda() {
  ensure_conda
  "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

base_run_docker() {
  local img_workspace img_conda_path workdir host_pwd rel_pwd

  set_image_vars
  command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
  img_workspace="$ENV_REPO_ROOT"
  img_conda_path="/home/hlavo/miniconda3/bin/conda"
  tty_arg="${tty_arg:-}"
  host_pwd="$(pwd -P)"
  rel_pwd="${host_pwd#$REPO_ROOT}"
  if [[ "$rel_pwd" != "$host_pwd" ]]; then
    workdir="${img_workspace}${rel_pwd}"
  else
    workdir="$img_workspace"
  fi

  docker run --rm \
    ${tty_arg} \
    -e HLAVO_UID="$(id -u)" \
    -e HLAVO_GID="$(id -g)" \
    -v "$REPO_ROOT:$img_workspace" \
    -w "$workdir" \
    "$IMAGE_REF" \
    "$img_conda_path" run -n "$ENV_NAME" "$@"
}

env_build() {
  base_build "$@"
  venv_overlay "$@"
}

run_cmd() {
  venv_ensure
  [[ $# -ge 1 ]] || die "Missing command."
  cmd="$(printf '%q ' "$@")"
  base_run bash -lc "source \"$VENV_DIR/bin/activate\"; exec $cmd"
}

case "$HLAVO_MODE" in
  conda)
    ENV_REPO_ROOT="$REPO_ROOT"
    VENV_DIR="$ENV_REPO_ROOT/dev/venv-conda"
    HOST_VENV_DIR="$VENV_DIR"
    base_run() { base_run_conda "$@"; }
    base_build() { base_build_conda "$@"; }
    ;;
  docker)
    ENV_REPO_ROOT="${ENV_REPO_ROOT:-/home/hlavo/workspace}"
    VENV_DIR="$ENV_REPO_ROOT/dev/venv-docker"
    HOST_VENV_DIR="$REPO_ROOT/dev/venv-docker"
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
  # Light test to see if venv exists on host
  # Can not test working python without calling container (slow)
  if [[ ! -d "$HOST_VENV_DIR" ]]; then
    venv_overlay
  fi
}

venv_overlay() {
  if [[ "${1:-}" == "-f" ]]; then
    rm -rf "$VENV_DIR"
  fi

  base_run bash -lc "
    [ -d "$VENV_DIR" ] || python -m venv --system-site-packages "$VENV_DIR"

    source \"$VENV_DIR/bin/activate\"
    python -m pip install --upgrade pip
    if [[ -f \"$ENV_REPO_ROOT/requirements.txt\" ]]; then
      python -m pip install -r \"$ENV_REPO_ROOT/requirements.txt\"
    fi
    if [[ -f \"$ENV_REPO_ROOT/pyproject.toml\" ]]; then
      python -m pip install -e \"$ENV_REPO_ROOT\"
    else
      echo \"Note: $ENV_REPO_ROOT/pyproject.toml not found — skipping editable install.\"
    fi
  "
}
