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
import inspect
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


def check_parflow_runtime() -> CheckResult:
    try:
        from parflow import Run  # type: ignore
    except Exception as e:
        return CheckResult("parflow", False, f"Failed to import Run: {type(e).__name__}: {e}")

    try:
        r = Run("smoke")

        # Minimal 1D domain setup
        r.FileVersion = 4
        r.Process.Topology.P = 1
        r.Process.Topology.Q = 1
        r.Process.Topology.R = 1

        r.ComputationalGrid.Lower.X = 0.0
        r.ComputationalGrid.Lower.Y = 0.0
        r.ComputationalGrid.Lower.Z = 0.0
        r.ComputationalGrid.DX = 1.0
        r.ComputationalGrid.DY = 1.0
        r.ComputationalGrid.DZ = 1.0
        r.ComputationalGrid.NX = 10
        r.ComputationalGrid.NY = 1
        r.ComputationalGrid.NZ = 1

        r.GeomInput.Names = "domain_input"
        r.GeomInput.domain_input.InputType = "Box"
        r.GeomInput.domain_input.GeomName = "domain"
        r.Geom.domain.Lower.X = 0.0
        r.Geom.domain.Lower.Y = 0.0
        r.Geom.domain.Lower.Z = 0.0
        r.Geom.domain.Upper.X = 10.0
        r.Geom.domain.Upper.Y = 1.0
        r.Geom.domain.Upper.Z = 1.0
        r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

        r.Geom.domain.Perm.Type = "Constant"
        r.Geom.domain.Perm.Value = 1.0
        r.Geom.domain.Porosity.Type = "Constant"
        r.Geom.domain.Porosity.Value = 0.3
        r.Geom.domain.SpecificStorage.Value = 1.0e-6

        r.Phase.Names = "water"
        r.Phase.water.Density.Type = "Constant"
        r.Phase.water.Density.Value = 1.0
        r.Phase.water.Viscosity.Type = "Constant"
        r.Phase.water.Viscosity.Value = 1.0

        r.Gravity = 1.0

        r.TimeStep.Type = "Constant"
        r.TimeStep.Value = 1.0
        r.TimingInfo.BaseUnit = 1.0
        r.TimingInfo.StartCount = 0
        r.TimingInfo.StartTime = 0.0
        r.TimingInfo.StopTime = 1.0
        r.TimingInfo.DumpInterval = 1.0

        r.Cycle.Names = "constant"
        r.Cycle.constant.Names = "alltime"
        r.Cycle.constant.alltime.Length = 1
        r.Cycle.constant.Repeat = -1

        r.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"
        r.Patch.Names = r.Geom.domain.Patches
        for patch in ["x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"]:
            bc = r.Patch[patch].BCPressure
            bc.Type = "FluxConst"
            bc.Cycle = "constant"
            bc.alltime.Value = 0.0

        r.ICPressure.Type = "Constant"
        r.ICPressure.GeomNames = "domain"
        r.ICPressure.Value = 0.0

        r.Phase.RelPerm.Type = "VanGenuchten"
        r.Phase.RelPerm.Alpha = 1.0
        r.Phase.RelPerm.N = 2.0
        r.Phase.Saturation.Type = "VanGenuchten"
        r.Phase.Saturation.Alpha = 1.0
        r.Phase.Saturation.N = 2.0
        r.Phase.Saturation.SRes = 0.1
        r.Phase.Saturation.SSat = 1.0

        r.Solver = "Richards"
        r.Solver.MaxIter = 1
        r.Solver.Nonlinear.MaxIter = 3
        r.Solver.Nonlinear.ResidualTol = 1.0e-6

        run_root = os.environ.get("HLAVO_WORKSPACE", "/home/hlavo/workspace")
        run_dir = os.path.join(run_root, "_parflow_smoke")
        os.makedirs(run_dir, exist_ok=True)

        sig = inspect.signature(r.run)
        kwargs: dict[str, object] = {}
        if "working_directory" in sig.parameters:
            kwargs["working_directory"] = run_dir
        r.run(**kwargs)
        return CheckResult("parflow", True, f"Run.run executed in {run_dir}")
    except Exception as e:
        return CheckResult("parflow", False, f"{type(e).__name__}: {e}")


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
        "parflow",
    ]
    for m in imports:
        checks.append(check_import(m))

    # ParFlow runtime check
    checks.append(check_parflow_runtime())

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
