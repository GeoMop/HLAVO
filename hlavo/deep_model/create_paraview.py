from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import flopy
from flopy.utils import postprocessing

from run_model import (
    RunConfig,
    _export_materials_to_paraview,
    _export_results_to_paraview,
    _groundwater_surface_from_head,
    _grid_arrays_from_npz,
    _material_class_from_model_data,
    _surface_to_nodes,
)

LOG = logging.getLogger(__name__)

def create_paraview(config_path: Path, workspace: Path | None = None) -> None:
    run_config = RunConfig.from_yaml(config_path, workspace)
    grid_data = _grid_arrays_from_npz(run_config.grid_output_path)
    model_data = _grid_arrays_from_npz(run_config.material_parameters_path)

    el_dims = np.asarray(grid_data["el_dims"], dtype=int)
    assert el_dims.size == 3, "el_dims must contain 3 values"
    nx = int(el_dims[0])
    ny = int(el_dims[1])
    nz = int(el_dims[2])

    active_mask = np.asarray(grid_data["active_mask"], dtype=bool)
    top = np.asarray(model_data["top"], dtype=float)
    botm = np.asarray(model_data["botm"], dtype=float)
    materials = np.asarray(model_data["materials"], dtype=int)
    idomain = np.asarray(model_data["idomain"], dtype=int)
    hk = np.asarray(model_data["kh"] if "kh" in model_data else model_data["hk"], dtype=float)
    material_class = _material_class_from_model_data(model_data, materials)

    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"
    assert top.shape == (ny, nx), "top shape mismatch"
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"
    assert materials.shape == (nz, ny, nx), "materials shape mismatch"
    assert idomain.shape == (nz, ny, nx), "idomain shape mismatch"
    assert hk.shape == (nz, ny, nx), "hk shape mismatch"

    workdir = run_config.workspace
    head_path = workdir / f"{run_config.sim_name}.hds"
    budget_path = workdir / f"{run_config.sim_name}.cbc"
    assert head_path.exists(), f"Head output not found: {head_path}"
    assert budget_path.exists(), f"Budget output not found: {budget_path}"

    hds = flopy.utils.binaryfile.HeadFile(str(head_path))
    times = hds.get_times()
    assert times, "No time steps found in head file"
    head_initial = np.asarray(hds.get_data(totim=times[0]), dtype=float)
    head = np.asarray(hds.get_data(totim=times[-1]), dtype=float)
    assert head.shape == (nz, ny, nx), "Head output shape mismatch"
    assert head_initial.shape == (nz, ny, nx), "Head output shape mismatch"
    groundwater_surface_initial = _groundwater_surface_from_head(head_initial, idomain, top, botm)
    groundwater_surface_final = _groundwater_surface_from_head(head, idomain, top, botm)
    groundwater_surface_change = groundwater_surface_final - groundwater_surface_initial

    sim = flopy.mf6.MFSimulation.load(sim_ws=str(workdir), verbosity_level=0)
    gwf = sim.get_model(run_config.sim_name)
    assert gwf is not None, f"Model {run_config.sim_name} not found in {workdir}"
    cbc = flopy.utils.CellBudgetFile(str(budget_path))
    spdis = cbc.get_data(text="SPDIS")
    assert spdis, "Specific discharge (SPDIS) not found in budget file"
    qx, qy, qz = postprocessing.get_specific_discharge(spdis[-1], gwf)

    _export_results_to_paraview(
        run_config.paraview_output_path,
        grid_data,
        head,
        hk,
        qx,
        qy,
        qz,
        idomain,
        materials,
        material_class,
        top,
        botm,
        run_config.paraview_quantities,
        run_config.paraview_include_inactive,
        run_config.paraview_z_scale,
        groundwater_surface=groundwater_surface_final,
        groundwater_surface_change=groundwater_surface_change,
    )
    _export_materials_to_paraview(
        run_config.paraview_materials_output_path,
        grid_data,
        materials,
        material_class,
        active_mask,
        top,
        botm,
        run_config.paraview_quantities,
        run_config.paraview_include_inactive,
        run_config.paraview_z_scale,
    )
    if run_config.paraview_surface_timeseries:
        _export_groundwater_surface_timeseries(
            run_config,
            grid_data,
            idomain,
            top,
            botm,
            hds,
            times,
        )
    LOG.info("Paraview export complete")


def _export_groundwater_surface_timeseries(
    run_config: RunConfig,
    grid_data: dict[str, np.ndarray],
    idomain: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
    hds: flopy.utils.binaryfile.HeadFile,
    times: list[float],
) -> None:
    import pyvista as pv

    x_nodes = np.asarray(grid_data["x_nodes"], dtype=float)
    y_nodes = np.asarray(grid_data["y_nodes"], dtype=float)
    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")

    output_dir = Path(run_config.paraview_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_entries: list[tuple[float, str]] = []

    for idx, time_value in enumerate(times):
        head = np.asarray(hds.get_data(totim=time_value), dtype=float)
        groundwater_surface = _groundwater_surface_from_head(head, idomain, top, botm)
        nodes = _surface_to_nodes(groundwater_surface)
        z_grid = nodes.T
        surface = pv.StructuredGrid(x_grid, y_grid, z_grid)
        surface.point_data["groundwater_surface"] = z_grid.ravel(order="F")
        filename = f"{run_config.sim_name}_gw_surface_t{idx:04d}.vts"
        output_path = output_dir / filename
        surface.save(str(output_path))
        series_entries.append((float(time_value), filename))

    pvd_path = output_dir / run_config.paraview_surface_timeseries_name
    _write_pvd(pvd_path, series_entries)
    LOG.info("Saved groundwater surface time series to %s", pvd_path)


def _write_pvd(path: Path, entries: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_value, file_name in entries:
        lines.append(f'    <DataSet timestep="{time_value}" group="" part="0" file="{file_name}"/>')
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Paraview outputs from finished Modflow run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.yaml"),
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override model workspace directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    create_paraview(args.config, args.workspace)


if __name__ == "__main__":
    main()
