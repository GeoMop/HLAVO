#!/usr/bin/env python3
"""
Basic environment smoke test for HLAVO stack.

Checks:
  - MODFLOW 6 executable is present and runnable (`mf6 -h`)
  - Imports key Python packages
  - Verifies cfgrib can read GRIB via ecCodes by creating a minimal GRIB2 message in-memory
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def run_cmd(cmd: list[str], timeout_s: int = 20) -> CheckResult:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        ok = p.returncode == 0
        detail = f"returncode={p.returncode}\nstdout:\n{out}\nstderr:\n{err}".strip()
        return CheckResult(" ".join(cmd), ok, detail)
    except FileNotFoundError:
        return CheckResult(" ".join(cmd), False, "command not found")
    except subprocess.TimeoutExpired:
        return CheckResult(" ".join(cmd), False, f"timed out after {timeout_s}s")


def check_mf6() -> CheckResult:
    mf6_path = shutil.which("mf6")
    if not mf6_path:
        return CheckResult("mf6", False, "mf6 not found on PATH")
    # `mf6 -h` sometimes returns non-zero depending on build; try -h then plain invocation.
    res_h = run_cmd(["mf6", "-h"])
    if res_h.ok:
        return CheckResult("mf6", True, f"found at {mf6_path}\n{res_h.detail}")
    # Try `mf6` (should print help/usage)
    res_plain = run_cmd(["mf6"])
    if res_plain.ok:
        return CheckResult("mf6", True, f"found at {mf6_path}\n{res_plain.detail}")
    return CheckResult("mf6", False, f"found at {mf6_path} but could not run successfully\n\n-h:\n{res_h.detail}\n\nplain:\n{res_plain.detail}")


def check_import(module_name: str, attr_version: Optional[str] = None) -> CheckResult:
    try:
        mod = importlib.import_module(module_name)
        ver = ""
        if attr_version and hasattr(mod, attr_version):
            ver = f" (version: {getattr(mod, attr_version)})"
        elif hasattr(mod, "__version__"):
            ver = f" (version: {getattr(mod, '__version__')})"
        return CheckResult(f"import {module_name}", True, f"imported OK{ver}")
    except Exception as e:
        return CheckResult(f"import {module_name}", False, f"{type(e).__name__}: {e}")


def check_cfgrib_roundtrip() -> CheckResult:
    """
    Create a minimal GRIB2 message via ecCodes and read it via cfgrib.

    This validates:
      - Python bindings for ecCodes are installed and working
      - ecCodes library is present at runtime
      - cfgrib can open GRIB files and talk to ecCodes
    """
    try:
        import eccodes  # type: ignore
    except Exception as e:
        return CheckResult("cfgrib/ecCodes roundtrip", False, f"Failed to import eccodes: {type(e).__name__}: {e}")

    try:
        import cfgrib  # noqa: F401
    except Exception as e:
        return CheckResult("cfgrib/ecCodes roundtrip", False, f"Failed to import cfgrib: {type(e).__name__}: {e}")

    # Build a tiny GRIB2 message.
    try:
        gid = eccodes.codes_grib_new_from_samples("GRIB2")  # minimal template
        # Set a few required-ish keys to make it sensible
        eccodes.codes_set(gid, "discipline", 0)  # meteorological products
        eccodes.codes_set(gid, "parameterCategory", 0)
        eccodes.codes_set(gid, "parameterNumber", 0)  # Temperature (often)
        eccodes.codes_set(gid, "typeOfFirstFixedSurface", 1)  # ground or water surface
        eccodes.codes_set(gid, "level", 0)
        eccodes.codes_set(gid, "dataDate", 20250101)
        eccodes.codes_set(gid, "dataTime", 0)

        # Define a tiny regular lat-lon grid 2x2
        eccodes.codes_set(gid, "gridType", "regular_ll")
        eccodes.codes_set(gid, "Ni", 2)
        eccodes.codes_set(gid, "Nj", 2)
        eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", 50.0)
        eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", 14.0)
        eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", 49.0)
        eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", 15.0)
        eccodes.codes_set(gid, "iDirectionIncrementInDegrees", 1.0)
        eccodes.codes_set(gid, "jDirectionIncrementInDegrees", 1.0)

        values = [273.15, 274.15, 275.15, 276.15]  # 2x2
        eccodes.codes_set_values(gid, values)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tiny.grib2")
            with open(path, "wb") as f:
                eccodes.codes_write(gid, f)
            eccodes.codes_release(gid)

            # Now read with cfgrib
            import cfgrib  # type: ignore

            ds = cfgrib.open_dataset(path, errors="raise")
            # cfgrib typically puts values in a variable named after shortName (e.g., 't')
            # We just check the dataset has some data variables and dimensions.
            if not ds.data_vars:
                return CheckResult("cfgrib/ecCodes roundtrip", False, "cfgrib opened dataset but found no data variables")
            return CheckResult(
                "cfgrib/ecCodes roundtrip",
                True,
                f"Opened GRIB with cfgrib. data_vars={list(ds.data_vars.keys())}, dims={dict(ds.dims)}",
            )
    except Exception as e:
        return CheckResult("cfgrib/ecCodes roundtrip", False, f"{type(e).__name__}: {e}")


def main() -> int:
    checks: list[CheckResult] = []

    # Executable check
    checks.append(check_mf6())

    # Imports (keep aligned with your project deps)
    imports = [
        "attrs",
        "yaml",         # PyYAML
        "numpy",
        "scipy",
        "filterpy",
        "matplotlib",
        "xarray",
        "dask",
        "distributed",
        "zarr",
        "flopy",
        "modflow_devtools",
        "requests",
        "cfgrib",
    ]
    for m in imports:
        checks.append(check_import(m))

    # cfgrib deep test
    checks.append(check_cfgrib_roundtrip())

    # Report
    print("\n=== HLAVO environment smoke test ===\n")
    ok_all = True
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"[{status}] {c.name}")
        if c.detail:
            print(f"  {c.detail.replace(chr(10), chr(10)+'  ')}")
        print()
        ok_all = ok_all and c.ok

    if ok_all:
        print("All checks passed ✅")
        return 0
    else:
        print("Some checks failed ❌")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
