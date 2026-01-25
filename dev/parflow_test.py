#!/usr/bin/env python3
"""
minimal_parflow_run.py

A minimal ParFlow simulation using the Python interface:
- Richards equation
- Small box domain (10 x 10 x 1)
- Constant permeability/porosity
- Van Genuchten relperm + saturation
- BCs: no-flow on sides, constant pressure at bottom/top
- One timestep, one dump

Run:
  python minimal_parflow_run.py

Requirements:
  - ParFlow installed (pfsol, pfrichards, etc. available via parflow run wrapper)
  - Python package `parflow` available (ParFlow Python tools)
"""

import os
from pathlib import Path

try:
    from parflow import Run
except Exception as e:
    raise SystemExit(
        "Could not import ParFlow Python interface.\n"
        "Make sure ParFlow + its python package are installed and on PYTHONPATH.\n"
        f"Import error: {e}"
    )


def build_run(name: str = "minimal"):
    run = Run(name, __file__)

    # -------------------------------------------------------------------------
    # Basic topology (single process)
    # -------------------------------------------------------------------------
    run.Process.Topology.P = 1
    run.Process.Topology.Q = 1
    run.Process.Topology.R = 1

    # -------------------------------------------------------------------------
    # Computational grid (10x10x1 cells, unit spacing)
    # -------------------------------------------------------------------------
    run.ComputationalGrid.Lower.X = 0.0
    run.ComputationalGrid.Lower.Y = 0.0
    run.ComputationalGrid.Lower.Z = 0.0

    run.ComputationalGrid.DX = 1.0
    run.ComputationalGrid.DY = 1.0
    run.ComputationalGrid.DZ = 1.0

    run.ComputationalGrid.NX = 10
    run.ComputationalGrid.NY = 10
    run.ComputationalGrid.NZ = 1

    # -------------------------------------------------------------------------
    # Geometry: single box domain
    # -------------------------------------------------------------------------
    run.GeomInput.Names = "domain_input"
    run.GeomInput.domain_input.InputType = "Box"
    run.GeomInput.domain_input.GeomName = "domain"

    run.Geom.domain.Lower.X = 0.0
    run.Geom.domain.Lower.Y = 0.0
    run.Geom.domain.Lower.Z = 0.0

    run.Geom.domain.Upper.X = 10.0
    run.Geom.domain.Upper.Y = 10.0
    run.Geom.domain.Upper.Z = 1.0

    run.Geom.domain.Patches = "left right front back bottom top"

    # -------------------------------------------------------------------------
    # Physics setup
    # -------------------------------------------------------------------------
    run.Solver = "Richards"
    run.Gravity = 1.0

    run.Phase.Names = "water"
    run.Phase.water.Density.Type = "Constant"
    run.Phase.water.Density.Value = 1.0
    run.Phase.water.Viscosity.Type = "Constant"
    run.Phase.water.Viscosity.Value = 1.0

    # -------------------------------------------------------------------------
    # Material properties (constant)
    # -------------------------------------------------------------------------
    run.Geom.Perm.Names = "domain"
    run.Geom.domain.Perm.Type = "Constant"
    run.Geom.domain.Perm.Value = 1.0

    run.SpecificStorage.Type = "Constant"
    run.SpecificStorage.GeomNames = "domain"
    run.Geom.domain.SpecificStorage.Value = 1.0e-4

    run.Geom.Porosity.GeomNames = "domain"
    run.Geom.domain.Porosity.Type = "Constant"
    run.Geom.domain.Porosity.Value = 0.35

    # Van Genuchten functions for saturation + relative permeability
    run.Phase.RelPerm.Type = "VanGenuchten"
    run.Phase.RelPerm.GeomNames = "domain"
    run.Geom.domain.RelPerm.Alpha = 2.0
    run.Geom.domain.RelPerm.N = 2.0

    run.Phase.Saturation.Type = "VanGenuchten"
    run.Phase.Saturation.GeomNames = "domain"
    run.Geom.domain.Saturation.Alpha = 2.0
    run.Geom.domain.Saturation.N = 2.0
    run.Geom.domain.Saturation.SRes = 0.1
    run.Geom.domain.Saturation.SSat = 1.0

    # -------------------------------------------------------------------------
    # Time control: 1 step (t=0 -> t=1), dump at t=1
    # -------------------------------------------------------------------------
    run.TimingInfo.BaseUnit = 1.0
    run.TimingInfo.StartCount = 0
    run.TimingInfo.StartTime = 0.0
    run.TimingInfo.StopTime = 1.0
    run.TimingInfo.DumpInterval = 1.0

    run.TimeStep.Type = "Constant"
    run.TimeStep.Value = 1.0

    # -------------------------------------------------------------------------
    # Cycles (required for BCs)
    # -------------------------------------------------------------------------
    run.Cycle.Names = "constant"
    run.Cycle.constant.Names = "alltime"
    run.Cycle.constant.alltime.Length = 1
    run.Cycle.constant.Repeat = -1

    # -------------------------------------------------------------------------
    # Initial condition: constant pressure everywhere
    # -------------------------------------------------------------------------
    run.ICPressure.Type = "Constant"
    run.ICPressure.GeomNames = "domain"
    run.Geom.domain.ICPressure.Value = -0.5

    # -------------------------------------------------------------------------
    # Boundary conditions
    #   - No-flow (FluxConst = 0) on left/right/front/back
    #   - Constant pressure at bottom/top (DirConst)
    # -------------------------------------------------------------------------
    run.BCPressure.PatchNames = "left right front back bottom top"

    for patch in ["left", "right", "front", "back"]:
        getattr(run.Patch, patch).BCPressure.Type = "FluxConst"
        getattr(run.Patch, patch).BCPressure.Cycle = "constant"
        getattr(run.Patch, patch).BCPressure.alltime.Value = 0.0

    run.Patch.bottom.BCPressure.Type = "DirConst"
    run.Patch.bottom.BCPressure.Cycle = "constant"
    run.Patch.bottom.BCPressure.alltime.Value = 0.0

    run.Patch.top.BCPressure.Type = "DirConst"
    run.Patch.top.BCPressure.Cycle = "constant"
    run.Patch.top.BCPressure.alltime.Value = -1.0

    # -------------------------------------------------------------------------
    # Solver controls (kept minimal but stable)
    # -------------------------------------------------------------------------
    run.Solver.MaxIter = 50

    run.Solver.Nonlinear.MaxIter = 15
    run.Solver.Nonlinear.ResidualTol = 1.0e-6
    run.Solver.Nonlinear.EtaChoice = "EtaConstant"
    run.Solver.Nonlinear.EtaValue = 1.0e-2

    run.Solver.Linear.KrylovDimension = 20
    run.Solver.Linear.MaxRestarts = 2
    run.Solver.Linear.Preconditioner = "PFMG"

    # Output options (a couple useful defaults)
    run.Solver.PrintSubsurf = True
    run.Solver.Drop = 1.0e-20

    return run


def main():
    # Run in a clean local directory
    out_dir = Path.cwd() / "parflow_minimal_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = build_run("minimal")

    # Write the .pfidb input deck (and any derived files)
    run.write(working_directory=str(out_dir))

    # Execute ParFlow
    # This requires ParFlow executables to be available to the Python interface.
    run.run(working_directory=str(out_dir))

    print("\nDone.")
    print(f"Outputs should be under: {out_dir}")
    print("Look for files like minimal.out.press.*.pfb (pressure) and log outputs.")


if __name__ == "__main__":
    main()
