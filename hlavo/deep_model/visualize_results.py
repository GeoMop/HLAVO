from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import flopy
from flopy.utils import postprocessing

from run_model import (
    RunConfig,
    _groundwater_surface_from_head,
    _export_materials_to_paraview,
    _export_results_to_paraview,
    _grid_arrays_from_npz,
    _write_plan_view_plots,
)

LOG = logging.getLogger(__name__)


def visualize_results(config_path: Path, workspace: Path | None = None) -> None:
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
    heads = [np.asarray(hds.get_data(totim=time_value), dtype=float) for time_value in times]
    assert all(item.shape == (nz, ny, nx) for item in heads), "Head output shape mismatch"
    head = heads[-1]

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
        top,
        botm,
        run_config.paraview_quantities,
        run_config.paraview_include_inactive,
        run_config.paraview_z_scale,
    )
    _export_materials_to_paraview(
        run_config.paraview_materials_output_path,
        grid_data,
        materials,
        active_mask,
        top,
        botm,
        run_config.paraview_quantities,
        run_config.paraview_include_inactive,
        run_config.paraview_z_scale,
    )
    _write_plan_view_plots(
        run_config,
        grid_data,
        active_mask,
        head,
        qx,
        qy,
        idomain,
        materials,
        top,
        botm,
    )
    _write_temporal_groundwater_plots(
        run_config,
        grid_data,
        idomain,
        top,
        botm,
        heads,
        times,
    )

    LOG.info("Visualization complete")


def _write_temporal_groundwater_plots(
    run_config: RunConfig,
    grid_data: dict[str, np.ndarray],
    idomain: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
    heads: list[np.ndarray],
    times: list[float],
) -> None:
    import matplotlib.pyplot as plt

    water_tables = [
        _groundwater_surface_from_head(head, idomain, top, botm)
        for head in heads
    ]
    wt_initial = water_tables[0]
    wt_final = water_tables[-1]
    wt_change = wt_final - wt_initial

    x_nodes = np.asarray(grid_data["x_nodes"], dtype=float)
    y_nodes = np.asarray(grid_data["y_nodes"], dtype=float)
    extent = [x_nodes[0], x_nodes[-1], y_nodes[-1], y_nodes[0]]

    plot_dir = Path(run_config.plot_output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    vabs = float(np.nanmax(np.abs(wt_change))) if np.isfinite(wt_change).any() else 1.0
    img = plt.imshow(
        wt_change,
        origin="upper",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
    )
    plt.title("Groundwater Surface Change (Final - Initial)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.colorbar(img, label="dH [m]")
    plt.tight_layout()
    change_path = plot_dir / run_config.plot_groundwater_change_name
    plt.savefig(change_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved groundwater change plot to %s", change_path)

    x_idx = run_config.plot_xsection_x_index if run_config.plot_xsection_x_index is not None else wt_final.shape[1] // 2
    y_idx = run_config.plot_xsection_y_index if run_config.plot_xsection_y_index is not None else wt_final.shape[0] // 2
    assert 0 <= x_idx < wt_final.shape[1], "xsection_x_index out of range"
    assert 0 <= y_idx < wt_final.shape[0], "xsection_y_index out of range"

    hydrograph = [wt[y_idx, x_idx] for wt in water_tables]
    plt.figure(figsize=(8, 4))
    plt.plot(times, hydrograph, "b-o", linewidth=1.2, markersize=3)
    plt.title(f"Groundwater Hydrograph at Cell (x={x_idx}, y={y_idx})")
    plt.xlabel("Simulation time [days]")
    plt.ylabel("Groundwater surface Z")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hydro_path = plot_dir / run_config.plot_hydrograph_name
    plt.savefig(hydro_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved hydrograph plot to %s", hydro_path)

    sample_ids = [0, len(times) // 2, len(times) - 1]
    sample_ids = sorted(set(sample_ids))
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_centers = 0.5 * (y_nodes[:-1] + y_nodes[1:])

    plt.figure(figsize=(10, 4))
    for idx in sample_ids:
        plt.plot(
            x_centers,
            water_tables[idx][y_idx, :],
            linewidth=1.5,
            label=f"t={times[idx]:.2f} d",
        )
    plt.plot(x_centers, top[y_idx, :], "k--", linewidth=1.0, label="terrain")
    plt.title(f"Groundwater Surface X-Section in Time (y index {y_idx})")
    plt.xlabel("X (local)")
    plt.ylabel("Z")
    plt.legend(loc="best")
    plt.tight_layout()
    x_times_path = plot_dir / run_config.plot_xsection_x_times_name
    plt.savefig(x_times_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved X-section time plot to %s", x_times_path)

    plt.figure(figsize=(10, 4))
    for idx in sample_ids:
        plt.plot(
            y_centers,
            water_tables[idx][:, x_idx],
            linewidth=1.5,
            label=f"t={times[idx]:.2f} d",
        )
    plt.plot(y_centers, top[:, x_idx], "k--", linewidth=1.0, label="terrain")
    plt.title(f"Groundwater Surface Y-Section in Time (x index {x_idx})")
    plt.xlabel("Y (local)")
    plt.ylabel("Z")
    plt.legend(loc="best")
    plt.tight_layout()
    y_times_path = plot_dir / run_config.plot_xsection_y_times_name
    plt.savefig(y_times_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved Y-section time plot to %s", y_times_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Paraview outputs and plots from finished Modflow run."
    )
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
    visualize_results(args.config, args.workspace)


if __name__ == "__main__":
    main()
