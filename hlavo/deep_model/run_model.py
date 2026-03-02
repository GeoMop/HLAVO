from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import struct
from pathlib import Path

import attrs
import numpy as np
import yaml

import flopy
from flopy.utils import postprocessing

from build_modflow_grid import build_modflow_grid, raster_to_full_grid
from qgis_reader import ModelInputs

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class RunConfig:
    config_path: Path
    workspace: Path
    sim_name: str
    exe_name: str
    recharge_rate: float
    drain_conductance: float
    conductivity_default: float
    conductivity_by_layer: dict[str, float] | None
    conductivities: tuple[float, ...] | None
    output_grid_path: Path
    paraview_output_path: Path
    plot_enabled: bool
    plot_output_dir: Path
    plot_dpi: int
    plot_active_mask_name: str
    plot_idomain_name: str
    plot_head_name: str
    plot_velocity_name: str
    plot_xsection_x_name: str
    plot_xsection_y_name: str
    plot_xsection_y_index: int | None
    plot_xsection_x_index: int | None

    @staticmethod
    def from_yaml(config_path: Path, workspace: Path | None = None) -> "RunConfig":
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict), "Config YAML must be a mapping"

        model_raw = raw.get("model", {})
        assert isinstance(model_raw, dict), "model config must be a mapping"

        sim_name = str(model_raw.get("sim_name", "uhelna"))
        exe_name = str(model_raw.get("exe_name", "mf6"))
        recharge_rate = float(model_raw.get("recharge_rate", 1e-4))
        drain_conductance = float(model_raw.get("drain_conductance", 1.0))
        conductivity_default = float(model_raw.get("conductivity_default", 1e-6))

        conductivity_by_layer = model_raw.get("conductivity_by_layer")
        if conductivity_by_layer is not None:
            assert isinstance(conductivity_by_layer, dict), "conductivity_by_layer must be a mapping"
            conductivity_by_layer = {
                str(key): float(value) for key, value in conductivity_by_layer.items()
            }

        conductivities = model_raw.get("conductivities")
        if conductivities is not None:
            assert isinstance(conductivities, (list, tuple)), "conductivities must be a list"
            conductivities = tuple(float(value) for value in conductivities)

        output_grid_path = Path(raw.get("grid_output_path", Path("model") / "grid_materials.npz"))

        if workspace is None:
            workspace = Path(model_raw.get("workspace", "model"))

        paraview_raw = model_raw.get("paraview_results_output", raw.get("paraview_results_output"))
        if paraview_raw:
            paraview_output_path = Path(str(paraview_raw))
        else:
            paraview_output_path = Path(workspace) / f"{sim_name}_results.vtr"

        plot_raw = raw.get("plots", {})
        assert isinstance(plot_raw, dict), "plots config must be a mapping"
        plot_enabled = bool(plot_raw.get("enabled", True))
        plot_output_dir = Path(plot_raw.get("output_dir", Path(workspace) / "plots"))
        plot_dpi = int(plot_raw.get("dpi", 150))
        plot_active_mask_name = str(plot_raw.get("active_mask_name", "grid_active_mask.png"))
        plot_idomain_name = str(plot_raw.get("idomain_name", "idomain_top.png"))
        plot_head_name = str(plot_raw.get("head_name", "head_groundplan.png"))
        plot_velocity_name = str(plot_raw.get("velocity_name", "velocity_groundplan.png"))
        plot_xsection_x_name = str(plot_raw.get("xsection_x_name", "materials_x_section.png"))
        plot_xsection_y_name = str(plot_raw.get("xsection_y_name", "materials_y_section.png"))
        plot_xsection_y_index = plot_raw.get("xsection_y_index")
        if plot_xsection_y_index is not None:
            plot_xsection_y_index = int(plot_xsection_y_index)
        plot_xsection_x_index = plot_raw.get("xsection_x_index")
        if plot_xsection_x_index is not None:
            plot_xsection_x_index = int(plot_xsection_x_index)

        return RunConfig(
            config_path=config_path,
            workspace=Path(workspace),
            sim_name=sim_name,
            exe_name=exe_name,
            recharge_rate=recharge_rate,
            drain_conductance=drain_conductance,
            conductivity_default=conductivity_default,
            conductivity_by_layer=conductivity_by_layer,
            conductivities=conductivities,
            output_grid_path=output_grid_path,
            paraview_output_path=paraview_output_path,
            plot_enabled=plot_enabled,
            plot_output_dir=plot_output_dir,
            plot_dpi=plot_dpi,
            plot_active_mask_name=plot_active_mask_name,
            plot_idomain_name=plot_idomain_name,
            plot_head_name=plot_head_name,
            plot_velocity_name=plot_velocity_name,
            plot_xsection_x_name=plot_xsection_x_name,
            plot_xsection_y_name=plot_xsection_y_name,
            plot_xsection_y_index=plot_xsection_y_index,
            plot_xsection_x_index=plot_xsection_x_index,
        )


def _layer_conductivities(
    layer_names: list[str],
    conductivity_by_layer: dict[str, float] | None,
    conductivities: tuple[float, ...] | None,
    default_value: float,
) -> np.ndarray:
    if conductivity_by_layer is not None:
        values = []
        for name in layer_names:
            assert name in conductivity_by_layer, f"Missing conductivity for layer {name}"
            values.append(float(conductivity_by_layer[name]))
        return np.asarray(values, dtype=float)

    if conductivities is not None:
        assert len(conductivities) == len(layer_names), (
            "conductivities length must match number of layers"
        )
        return np.asarray(conductivities, dtype=float)

    LOG.warning("No layer conductivities provided, using default %s", default_value)
    return np.full(len(layer_names), float(default_value), dtype=float)


def _grid_arrays_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    assert npz_path.exists(), f"Grid NPZ not found: {npz_path}"
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _vtk_cell_array(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 3, "Cell array must be 3D"
    cell_values = np.transpose(values, (2, 1, 0))
    return cell_values.ravel(order="F")


def _vtk_cell_vectors(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    assert qx.shape == qy.shape == qz.shape, "Vector component shape mismatch"
    qx_flat = _vtk_cell_array(qx)
    qy_flat = _vtk_cell_array(qy)
    qz_flat = _vtk_cell_array(qz)
    return np.column_stack([qx_flat, qy_flat, qz_flat])


def _export_results_to_paraview(
    output_path: Path,
    grid_data: dict[str, np.ndarray],
    head: np.ndarray,
    hk: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    idomain: np.ndarray,
    materials: np.ndarray,
) -> Path:
    import pyvista as pv

    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)
    z_nodes = grid_data["z_nodes"].astype(float)

    assert head.shape == hk.shape == idomain.shape, "Head, hk, and idomain shape mismatch"
    assert qx.shape == qy.shape == qz.shape == head.shape, "Velocity shape mismatch"
    assert materials.shape == head.shape, "Materials shape mismatch"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = pv.RectilinearGrid(x_nodes, y_nodes, z_nodes)

    head_masked = np.where(idomain > 0, head, np.nan)
    qx_masked = np.where(idomain > 0, qx, np.nan)
    qy_masked = np.where(idomain > 0, qy, np.nan)
    qz_masked = np.where(idomain > 0, qz, np.nan)
    qmag = np.sqrt(qx_masked**2 + qy_masked**2 + qz_masked**2)
    grid.cell_data["head"] = _vtk_cell_array(head_masked[::-1])
    grid.cell_data["hk"] = _vtk_cell_array(hk[::-1])
    grid.cell_data["idomain"] = _vtk_cell_array(idomain[::-1].astype(np.int8))
    grid.cell_data["materials"] = _vtk_cell_array(materials.astype(np.int16))
    grid.cell_data["q"] = _vtk_cell_vectors(qx_masked[::-1], qy_masked[::-1], qz_masked[::-1])
    grid.cell_data["qmag"] = _vtk_cell_array(qmag[::-1])

    grid.save(str(output_path))
    assert output_path.exists(), f"Failed to write Paraview results: {output_path}"
    LOG.info("Saved Paraview results to %s", output_path)
    return output_path


def _write_plan_view_plots(
    run_config: RunConfig,
    grid_data: dict[str, np.ndarray],
    active_mask: np.ndarray,
    head: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    idomain: np.ndarray,
    materials: np.ndarray,
    top: np.ndarray,
) -> None:
    if not run_config.plot_enabled:
        return

    plot_dir = Path(run_config.plot_output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)
    extent = [x_nodes[0], x_nodes[-1], y_nodes[-1], y_nodes[0]]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(active_mask, origin="upper", extent=extent, cmap="Greys")
    plt.title("Active Cells (Plan View)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.tight_layout()
    active_path = plot_dir / run_config.plot_active_mask_name
    plt.savefig(active_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved active mask plot to %s", active_path)

    idomain_top = (idomain[0] > 0).astype(int)
    plt.figure(figsize=(8, 6))
    plt.imshow(idomain_top, origin="upper", extent=extent, cmap="Greys")
    plt.title("IDOMAIN (Top Layer, Plan View)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.tight_layout()
    idomain_path = plot_dir / run_config.plot_idomain_name
    plt.savefig(idomain_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved idomain plot to %s", idomain_path)

    head_plan = np.where(idomain[0] > 0, head[0], np.nan)
    plt.figure(figsize=(8, 6))
    img = plt.imshow(head_plan, origin="upper", extent=extent, cmap="viridis")
    plt.title("Hydraulic Head (Top Layer, Plan View)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.colorbar(img, label="Head")
    plt.tight_layout()
    head_path = plot_dir / run_config.plot_head_name
    plt.savefig(head_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved head plot to %s", head_path)

    velocity_plan = np.where(
        idomain[0] > 0, np.sqrt(qx[0] ** 2 + qy[0] ** 2), np.nan
    )
    plt.figure(figsize=(8, 6))
    img = plt.imshow(velocity_plan, origin="upper", extent=extent, cmap="magma")
    plt.title("Flow Velocity Magnitude (Top Layer, Plan View)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.colorbar(img, label="Velocity")
    plt.tight_layout()
    velocity_path = plot_dir / run_config.plot_velocity_name
    plt.savefig(velocity_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved velocity plot to %s", velocity_path)

    _write_cross_section_plots(
        run_config,
        grid_data,
        materials,
        top,
    )


def _write_cross_section_plots(
    run_config: RunConfig,
    grid_data: dict[str, np.ndarray],
    materials: np.ndarray,
    top: np.ndarray,
) -> None:
    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)
    z_step = float(grid_data["step"][2])
    nz, ny, nx = materials.shape

    y_index = run_config.plot_xsection_y_index
    if y_index is None:
        y_index = ny // 2
    assert 0 <= y_index < ny, "xsection_y_index out of range"

    x_index = run_config.plot_xsection_x_index
    if x_index is None:
        x_index = nx // 2
    assert 0 <= x_index < nx, "xsection_x_index out of range"

    import matplotlib.pyplot as plt

    # X-direction cross-section at fixed Y index.
    materials_x = materials[:, y_index, :]
    top_line_x = top[y_index, :]
    z_edges_center = top_line_x[None, :] - z_step * np.arange(nz + 1, dtype=float)[:, None]
    z_edges = np.empty((nz + 1, nx + 1), dtype=float)
    z_edges[:, 1:-1] = 0.5 * (z_edges_center[:, :-1] + z_edges_center[:, 1:])
    z_edges[:, 0] = z_edges_center[:, 0]
    z_edges[:, -1] = z_edges_center[:, -1]
    x_edges = np.tile(x_nodes[None, :], (nz + 1, 1))

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_edges, z_edges, materials_x, shading="auto", cmap="tab20")
    plt.gca().invert_yaxis()
    plt.title(f"Materials X-Section (y index {y_index})")
    plt.xlabel("X (local)")
    plt.ylabel("Z")
    plt.tight_layout()
    xsec_path = Path(run_config.plot_output_dir) / run_config.plot_xsection_x_name
    plt.savefig(xsec_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved X-section plot to %s", xsec_path)

    # Y-direction cross-section at fixed X index.
    materials_y = materials[:, :, x_index]
    top_line_y = top[:, x_index]
    z_edges_center = top_line_y[None, :] - z_step * np.arange(nz + 1, dtype=float)[:, None]
    z_edges = np.empty((nz + 1, ny + 1), dtype=float)
    z_edges[:, 1:-1] = 0.5 * (z_edges_center[:, :-1] + z_edges_center[:, 1:])
    z_edges[:, 0] = z_edges_center[:, 0]
    z_edges[:, -1] = z_edges_center[:, -1]
    y_edges = np.tile(y_nodes[None, :], (nz + 1, 1))

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(y_edges, z_edges, materials_y, shading="auto", cmap="tab20")
    plt.gca().invert_yaxis()
    plt.title(f"Materials Y-Section (x index {x_index})")
    plt.xlabel("Y (local)")
    plt.ylabel("Z")
    plt.tight_layout()
    ysec_path = Path(run_config.plot_output_dir) / run_config.plot_xsection_y_name
    plt.savefig(ysec_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved Y-section plot to %s", ysec_path)


def _materials_from_rasters_per_column(
    rasters: tuple["RasterLayer", ...],
    grid: "Grid",
    active_mask: np.ndarray,
    top: np.ndarray,
) -> np.ndarray:
    ny = int(grid.el_dims[1])
    nx = int(grid.el_dims[0])
    nz = int(grid.el_dims[2])
    z_step = float(grid.step[2])

    z_centers = top[None, :, :] - z_step * (np.arange(nz, dtype=float)[:, None, None] + 0.5)
    materials = np.full((nz, ny, nx), -1, dtype=int)

    last_top = top - z_step * nz
    last_id = np.zeros((ny, nx), dtype=int)
    last_id[~active_mask] = -1

    rasters_bottom_up = tuple(reversed(rasters))
    for i_layer, raster in enumerate(rasters_bottom_up):
        layer_full = raster_to_full_grid(raster, grid)
        layer_data = np.ma.filled(layer_full, np.nan)
        layer_mask = np.ma.getmaskarray(layer_full)

        valid = active_mask & (~layer_mask) & (layer_data > last_top)
        if not np.any(valid):
            LOG.debug("Layer %s has no valid updates", raster.name)
            continue

        condition = (
            (z_centers >= last_top[None, :, :])
            & (z_centers < layer_data[None, :, :])
            & valid[None, :, :]
        )
        materials = np.where(condition, last_id[None, :, :], materials)

        last_top[valid] = layer_data[valid]
        last_id[valid] = i_layer
        LOG.debug(
            "Assigned layer %s (index %s) to %s cells (per-column)",
            raster.name,
            i_layer,
            int(condition.sum()),
        )

    return materials


def _resolve_executable(exe_name: str) -> Path | None:
    candidate = Path(exe_name)
    if candidate.is_file():
        return candidate
    resolved = shutil.which(exe_name)
    return Path(resolved) if resolved else None


def _elf_machine(path: Path) -> int | None:
    with path.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            return None
        handle.seek(18)
        return struct.unpack("<H", handle.read(2))[0]


def _assert_mf6_compatible(exe_path: Path) -> None:
    arch = platform.machine().lower()
    expected = {"x86_64": 62, "amd64": 62, "aarch64": 183}.get(arch)
    if expected is None:
        LOG.warning("Unknown architecture %s; skipping mf6 compatibility check", arch)
        return

    elf_machine = _elf_machine(exe_path)
    if elf_machine is None:
        return

    assert elf_machine == expected, (
        f"mf6 binary at {exe_path} is not compatible with {arch}. "
        "Install a native mf6 and set model.exe_name to its path "
        "(see tools/install_mf6.py)."
    )


def build_and_run(config_path: Path, workspace: Path | None = None) -> None:
    run_config = RunConfig.from_yaml(config_path, workspace)
    mpl_dir = run_config.workspace / ".mplconfig"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    exe_path = _resolve_executable(run_config.exe_name)
    assert exe_path is not None, (
        f"mf6 executable not found: {run_config.exe_name}. "
        "Set model.exe_name in the config to the full path."
    )
    _assert_mf6_compatible(exe_path)

    model_inputs = ModelInputs.from_yaml(run_config.config_path)
    grid_output_path = build_modflow_grid(run_config.config_path, run_config.output_grid_path)
    grid_data = _grid_arrays_from_npz(grid_output_path)

    grid = model_inputs.grid
    nx = int(grid.el_dims[0])
    ny = int(grid.el_dims[1])
    nz = int(grid.el_dims[2])

    active_mask = grid_data["active_mask"].astype(bool)
    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"

    z_nodes = grid_data["z_nodes"].astype(float)
    assert z_nodes.size == nz + 1, "z_nodes size mismatch"

    top_raster = model_inputs.rasters[0]
    top_full = raster_to_full_grid(top_raster, grid)
    top = np.ma.filled(top_full, z_nodes[-1]).astype(float)

    z_step = float(grid.step[2])
    # Build per-column botm from local top so layer 1 always has thickness z_step.
    # This avoids deactivating columns when the global botm sits above the local top.
    botm = top[None, :, :] - z_step * (np.arange(1, nz + 1, dtype=float)[:, None, None])

    idomain = np.broadcast_to(active_mask, (nz, ny, nx)).astype(int)
    materials = _materials_from_rasters_per_column(
        model_inputs.rasters, grid, active_mask, top
    )
    assert materials.shape == (nz, ny, nx), "materials shape mismatch"

    layer_names = [r.name for r in reversed(model_inputs.rasters)]
    conductivity_values = _layer_conductivities(
        layer_names,
        run_config.conductivity_by_layer,
        run_config.conductivities,
        run_config.conductivity_default,
    )

    materials_mf = materials[::-1, :, :]
    hk = np.full((nz, ny, nx), run_config.conductivity_default, dtype=float)
    valid = materials_mf >= 0
    hk[valid] = conductivity_values[materials_mf[valid]]

    workdir = run_config.workspace
    workdir.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(
        sim_name=run_config.sim_name,
        exe_name=str(exe_path),
        sim_ws=str(workdir),
    )

    flopy.mf6.ModflowTdis(sim, time_units="DAYS", perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=run_config.sim_name, save_flows=True)

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nz,
        nrow=ny,
        ncol=nx,
        delr=float(grid.step[0]),
        delc=float(grid.step[1]),
        top=top,
        botm=botm,
        idomain=idomain,
    )

    # MF6 expects starting heads per layer; use layer tops for a consistent 3D array.
    layer_tops = np.empty_like(botm)
    layer_tops[0] = top
    if nz > 1:
        layer_tops[1:] = botm[:-1]
    flopy.mf6.ModflowGwfic(gwf, strt=layer_tops)

    icelltype = np.zeros(nz, dtype=int)
    icelltype[0] = 1
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=hk, save_specific_discharge=True)

    recharge = np.where(active_mask, run_config.recharge_rate, 0.0)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)

    top_active = active_mask & (idomain[0] > 0)
    top_rows, top_cols = np.where(top_active)
    drain_cells = []
    for row, col in zip(top_rows.tolist(), top_cols.tolist()):
        drain_cells.append((0, int(row), int(col), float(top[row, col]), run_config.drain_conductance))

    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drain_cells)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{run_config.sim_name}.hds",
        budget_filerecord=f"{run_config.sim_name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    sim.write_simulation()
    success, buffer = sim.run_simulation()
    assert success, "Modflow6 did not terminate successfully"
    LOG.info("Modflow6 run complete")
    if buffer:
        LOG.debug("Modflow6 output: %s", "\n".join(buffer))

    head_path = workdir / f"{run_config.sim_name}.hds"
    assert head_path.exists(), f"Head output not found: {head_path}"
    head = flopy.utils.binaryfile.HeadFile(str(head_path)).get_data()
    assert head.shape == (nz, ny, nx), "Head output shape mismatch"
    budget_path = workdir / f"{run_config.sim_name}.cbc"
    assert budget_path.exists(), f"Budget output not found: {budget_path}"
    cbc = flopy.utils.CellBudgetFile(str(budget_path))
    spdis = cbc.get_data(text="SPDIS")
    assert spdis, "Specific discharge (SPDIS) not found in budget file"
    qx, qy, qz = postprocessing.get_specific_discharge(spdis[0], gwf)
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grid and run Modflow6.")
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
    build_and_run(args.config, args.workspace)


if __name__ == "__main__":
    main()
