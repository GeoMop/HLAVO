# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

# ---- Configurable versions ----
ARG NODE_MAJOR=22
ARG MF6_VERSION=6.6.3
# sha256 for mf6.6.3_linux.zip (update if you change MF6_VERSION)
ARG MF6_SHA256=4a2c98906f79403d7b752bc54e15042ebfcd794facc174d3309a6f871ce370fc
#ARG MF6_SHA256=fe4ff8055c8a55875f604937d7c7d25f9a5a147bee602e0887497f9e36580793

# ---- Base OS deps (add more if your requirements need them) ----
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      git \
      gnupg \
      # common build deps for Python packages with native extensions
      build-essential \
      pkg-config \
      # runtime libs often needed by scientific/Fortran-built binaries
      libgomp1 \
    ; \
    rm -rf /var/lib/apt/lists/*

# ---- Install Node.js (for npm) + Codex CLI ----
RUN set -eux; \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    npm i -g @openai/codex; \
    rm -rf /var/lib/apt/lists/*

# ---- Install MODFLOW 6 (mf6) ----
RUN set -eux; \
    curl -fL -o /tmp/mf6.zip \
      "https://github.com/MODFLOW-ORG/modflow6/releases/download/6.6.3/mf${MF6_VERSION}_linux.zip"; \
    echo "${MF6_SHA256}  /tmp/mf6.zip" | sha256sum -c -; \
    mkdir -p /opt; \
    unzip -q /tmp/mf6.zip -d /opt; \
    ls -l /opt/mf${MF6_VERSION}_linux/bin; \
    MF6_BIN="$(find /opt -type f -name mf6 -perm -111 | head -n 1)"; \
    test -n "${MF6_BIN}"; \
    install -m 0755 "${MF6_BIN}" /usr/local/bin/mf6; \
    rm -f /tmp/mf6.zip; \
    mf6 -h >/dev/null || true

# QGIS install
# RUN set -eux; \
#     apt-get update; \
#     apt-get install -y --no-install-recommends \
#       ca-certificates \
#       curl \
#       unzip \
#       git \
#       gnupg \
#       build-essential \
#       pkg-config \
#       libgomp1 \
#       \
#       # ---- Add these for QGIS/PyQGIS ----
#       python3 \
#       python3-pip \
#       qgis \
#       python3-qgis \
#     ; \
#     rm -rf /var/lib/apt/lists/*
#     
# ENV QGIS_PREFIX_PATH=/usr \
#     PYTHONPATH=/usr/share/qgis/python:/usr/lib/python3/dist-packages
# # Optional but often helpful in headless containers:
# ENV QT_QPA_PLATFORM=offscreen
    
# ---- Python dependencies ----
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN set -eux; \
    python -m pip install --upgrade pip; \
    pip install -r /workspace/requirements.txt

    
RUN set -eux; \
    mkdir -p /home/jb/.codex; \
    chmod 0775 /home/jb; \
    chmod 0775 /home/jb/.codex
    
# Optional: keep Codex config in a predictable place
# ENV CODEX_HOME=/root/.codex

CMD ["bash"]
