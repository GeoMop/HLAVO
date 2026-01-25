#!/usr/bin/env bash
set -xeuo pipefail

PREFIX="${1:-/opt/parflow_install}"
BIN_DIR="${PREFIX}/bin"
OUT_DIR="${PREFIX}/lib/ldd"

mkdir -p "$OUT_DIR"
: > "${OUT_DIR}/ldd.log"
: > "${OUT_DIR}/ldd.missing"
: > "${OUT_DIR}/ldd.libs"

for bin in "${BIN_DIR}"/*; do
  if [[ -x "$bin" ]]; then
    echo "== ${bin} ==" | tee -a "${OUT_DIR}/ldd.log"
    if command -v file >/dev/null 2>&1; then
      file "$bin" | tee -a "${OUT_DIR}/ldd.log" || true
    fi
    if ldd -v "$bin" > "${OUT_DIR}/ldd.tmp" 2>/dev/null; then
      cat "${OUT_DIR}/ldd.tmp" | tee -a "${OUT_DIR}/ldd.log"
    else
      echo "ldd failed for ${bin}" | tee -a "${OUT_DIR}/ldd.log"
      continue
    fi
    cat "${OUT_DIR}/ldd.tmp" \
      | awk '{ if ($2 == "=>") print $3; else if ($1 ~ /^\//) print $1; }' \
      | sort -u >> "${OUT_DIR}/ldd.libs"
    cat "${OUT_DIR}/ldd.tmp" | awk '/not found/{print $1}' >> "${OUT_DIR}/ldd.missing"
  fi
done

sort -u "${OUT_DIR}/ldd.libs" -o "${OUT_DIR}/ldd.libs"
sort -u "${OUT_DIR}/ldd.missing" -o "${OUT_DIR}/ldd.missing"

while read -r lib; do
  [[ -z "$lib" ]] && continue
  real="$(readlink -f "$lib" || true)"
  if [[ -n "$real" && -f "$real" ]]; then
    cp -v --dereference "$real" "$OUT_DIR/"
  else
    echo "$lib" >> "${OUT_DIR}/ldd.missing"
  fi
done < "${OUT_DIR}/ldd.libs"

sort -u "${OUT_DIR}/ldd.missing" -o "${OUT_DIR}/ldd.missing"

# Create SONAME symlinks (e.g., libfoo.so.1 -> libfoo.so.1.2.3)
for f in "${OUT_DIR}"/*.so.*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  if [[ "$base" =~ ^(lib[^/]+\.so\.[0-9]+) ]]; then
    ln -sf "$base" "${OUT_DIR}/${BASH_REMATCH[1]}"
  fi
done

ls -la "$OUT_DIR"
