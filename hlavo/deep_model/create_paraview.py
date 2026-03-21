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
    _grid_arrays_from_npz,
)

LOG = logging.getLogger(__name__)


def _material_class_from_interfaces(
    materials: np.ndarray,
    layer_names: list[str],
) -> np.ndarray:
    assert materials.ndim == 3, "materials must be 3D"
    class_by_layer = np.zeros(len(layer_names), dtype=np.int16)  # 0=other, 1=sand, 2=clay
    for idx, layer_name in enumerate(layer_names):
        if layer_name.startswith("Q") and layer_name.endswith("_base"):
            class_by_layer[idx] = 1
        elif layer_name.startswith("Q") and layer_name.endswith("_top"):
            class_by_layer[idx] = 2

    classes = np.full(materials.shape, -1, dtype=np.int16)  # -1=inactive/undefined
    valid = materials >= 0
    classes[valid] = class_by_layer[materials[valid]]
    return classes


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
    layer_names = [str(name) for name in model_data["layer_names"].tolist()]
    idomain = np.asarray(model_data["idomain"], dtype=int)
    hk = np.asarray(model_data["kh"] if "kh" in model_data else model_data["hk"], dtype=float)
    material_class = _material_class_from_interfaces(materials, layer_names)

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
    head = np.asarray(hds.get_data(totim=times[-1]), dtype=float)
    assert head.shape == (nz, ny, nx), "Head output shape mismatch"

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
    LOG.info("Paraview export complete")


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
