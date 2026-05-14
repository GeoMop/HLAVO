from __future__ import annotations

import logging
from pathlib import Path

import attrs
import flopy
import numpy as np
import xarray as xr

import hlavo.deep_model.model_3d_cfg as cfg3d
from hlavo.deep_model.pumping_well import PumpingWell
from hlavo.deep_model.qgis_reader import ModelGeometry

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class MaterialFields:
    top: np.ndarray
    botm: np.ndarray
    materials: np.ndarray
    materials_mf: np.ndarray
    active_mask: np.ndarray
    idomain: np.ndarray
    kh: np.ndarray
    kv: np.ndarray
    porosity: np.ndarray
    vg_alpha: np.ndarray
    vg_n: np.ndarray
    specific_yield: float
    specific_storage: float
    recharge_rate: float
    initial_head_offset: float
    layer_names: tuple[str, ...]


@attrs.define(frozen=True)
class SimulationBuildResult:
    sim: flopy.mf6.MFSimulation
    gwf: flopy.mf6.ModflowGwf
    material_fields: MaterialFields


def _init_value(material_dataset: xr.Dataset, variable: str, material_name: str) -> float:
    source_name = material_name if material_name in material_dataset.coords["material"].values else "all"
    values = material_dataset[variable].sel(material=source_name)
    if "bound" in values.dims:
        return float(values.sel(bound="init").item())
    return float(values.item())


def _init_bool(material_dataset: xr.Dataset, variable: str) -> bool:
    return bool(material_dataset[variable].sel(material="all").item())


def _init_int(material_dataset: xr.Dataset, variable: str) -> int:
    return int(material_dataset[variable].sel(material="all").item())


def _layer_material_name(layer_name: str) -> str:
    if layer_name.startswith("Q") and layer_name.endswith("_base"):
        return "sand"
    if layer_name.startswith("Q") and layer_name.endswith("_top"):
        return "clay"
    return "all"


def _layer_values(
    layer_names: tuple[str, ...],
    material_dataset: xr.Dataset,
    variable: str,
) -> np.ndarray:
    return np.asarray(
        [_init_value(material_dataset, variable, _layer_material_name(layer_name)) for layer_name in layer_names],
        dtype=float,
    )


def build_material_fields(*, geometry: ModelGeometry, material_dataset: xr.Dataset) -> MaterialFields:
    materials = np.asarray(geometry.materials, dtype=int)
    active_mask = np.asarray(geometry.active_mask, dtype=bool)
    top = np.asarray(geometry.top, dtype=float)
    botm = np.asarray(geometry.botm, dtype=float)
    layer_names = tuple(str(name) for name in geometry.layer_names)

    assert materials.ndim == 3, "geometry.materials must be 3D"
    nz, ny, nx = materials.shape
    assert active_mask.shape == (ny, nx), "geometry.active_mask shape mismatch"
    assert top.shape == (ny, nx), "geometry.top shape mismatch"
    assert botm.shape == (nz, ny, nx), "geometry.botm shape mismatch"

    idomain = np.broadcast_to(active_mask, (nz, ny, nx)).astype(int)
    materials_mf = materials[::-1, :, :]
    valid = materials_mf >= 0

    kh_by_layer = _layer_values(layer_names, material_dataset, "horizontal_conductivity")
    kv_by_layer = _layer_values(layer_names, material_dataset, "vertical_conductivity")

    kh = np.full((nz, ny, nx), _init_value(material_dataset, "horizontal_conductivity", "all"), dtype=float)
    kv = np.full((nz, ny, nx), _init_value(material_dataset, "vertical_conductivity", "all"), dtype=float)
    porosity = np.full((nz, ny, nx), _init_value(material_dataset, "porosity", "all"), dtype=float)
    vg_alpha = np.full((nz, ny, nx), _init_value(material_dataset, "vG_alpha", "all"), dtype=float)
    vg_n = np.full((nz, ny, nx), _init_value(material_dataset, "vG_n", "all"), dtype=float)

    kh[valid] = kh_by_layer[materials_mf[valid]]
    kv[valid] = kv_by_layer[materials_mf[valid]]

    return MaterialFields(
        top=top,
        botm=botm,
        materials=materials,
        materials_mf=materials_mf,
        active_mask=active_mask,
        idomain=idomain,
        kh=kh,
        kv=kv,
        porosity=porosity,
        vg_alpha=vg_alpha,
        vg_n=vg_n,
        specific_yield=_init_value(material_dataset, "specific_yield", "all"),
        specific_storage=_init_value(material_dataset, "specific_storage", "all"),
        recharge_rate=_init_value(material_dataset, "recharge_rate", "all"),
        initial_head_offset=_init_value(material_dataset, "initial_head_offset", "all"),
        layer_names=layer_names,
    )


def append_grid_lonlat_to_nam(
    nam_path: Path,
    grid_corners_lonlat: np.ndarray,
    epsg: int = 4326,
) -> None:
    assert grid_corners_lonlat.shape == (2, 2), "grid_corners_lonlat must be 2x2"
    sw = grid_corners_lonlat[0]
    ne = grid_corners_lonlat[1]
    marker = "Grid SW corner lon/lat"
    lines = [
        f"# {marker} (EPSG:{epsg}): {sw[1]:.6f}, {sw[0]:.6f}",
        f"# Grid NE corner lon/lat (EPSG:{epsg}): {ne[1]:.6f}, {ne[0]:.6f}",
    ]
    content = nam_path.read_text(encoding="utf-8")
    if marker in content:
        return
    existing = content.splitlines()
    insert_at = 1 if existing and existing[0].startswith("#") else 0
    updated = existing[:insert_at] + lines + existing[insert_at:]
    ending = "\n" if content.endswith("\n") else ""
    nam_path.write_text("\n".join(updated) + ending, encoding="utf-8")


def build_modflow_simulation(
    *,
    common: cfg3d.Model3DCommonConfig,
    geometry: ModelGeometry,
    material_dataset: xr.Dataset,
    workspace: Path,
    pumping_wells: tuple[PumpingWell, ...] = (),
    exe_name: str | Path | None = None,
) -> SimulationBuildResult:
    fields = build_material_fields(geometry=geometry, material_dataset=material_dataset)
    workdir = Path(workspace)
    workdir.mkdir(parents=True, exist_ok=True)
    simulation_exe = str(exe_name if exe_name is not None else common.exe_name)

    sim = flopy.mf6.MFSimulation(
        sim_name=common.sim_name,
        exe_name=simulation_exe,
        sim_ws=str(workdir),
    )
    perioddata = [(float(days), 1, 1.0) for days in common.stress_periods_days]
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(perioddata),
        perioddata=perioddata,
    )
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    nz, ny, nx = fields.idomain.shape
    step = np.asarray(geometry.grid.step, dtype=float)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=common.sim_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nz,
        nrow=ny,
        ncol=nx,
        delr=float(step[0]),
        delc=float(step[1]),
        top=fields.top,
        botm=fields.botm,
        idomain=fields.idomain,
    )

    layer_tops = np.empty_like(fields.botm)
    layer_tops[0] = fields.top
    if nz > 1:
        layer_tops[1:] = fields.botm[:-1]
    if fields.initial_head_offset != 0.0:
        layer_tops = layer_tops + fields.initial_head_offset
    flopy.mf6.ModflowGwfic(gwf, strt=layer_tops)

    icelltype = np.zeros(nz, dtype=int)
    icelltype[0] = 1
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=icelltype,
        ss=fields.specific_storage,
        sy=fields.specific_yield,
        transient={iper: True for iper in range(len(common.stress_periods_days))},
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=fields.kh, k33=fields.kv, save_specific_discharge=True)

    if common.recharge_series_m_per_day is not None:
        recharge_rates = common.recharge_series_m_per_day
    else:
        recharge_rates = tuple(fields.recharge_rate for _ in common.stress_periods_days)
    recharge_spd = {
        iper: np.where(fields.active_mask, float(rate), 0.0)
        for iper, rate in enumerate(recharge_rates)
    }
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge_spd)

    top_active = fields.active_mask & (fields.idomain[0] > 0)
    top_rows, top_cols = np.where(top_active)
    drain_cells = [
        (0, int(row), int(col), float(fields.top[row, col]), common.drain_conductance)
        for row, col in zip(top_rows.tolist(), top_cols.tolist())
    ]
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drain_cells)

    if pumping_wells:
        stress_period_data: dict[int, list[tuple[int, int, int, float]]] = {
            iper: [] for iper in range(len(common.stress_periods_days))
        }
        for pumping_well in pumping_wells:
            for iper, entries in pumping_well.stress_period_data().items():
                stress_period_data[iper].extend(entries)
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=stress_period_data, pname="WEL-1")

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{common.sim_name}.hds",
        budget_filerecord=f"{common.sim_name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    sim.write_simulation()
    append_grid_lonlat_to_nam(workdir / "mfsim.nam", geometry.grid_corners_lonlat, geometry.lonlat_epsg)
    append_grid_lonlat_to_nam(
        workdir / f"{common.sim_name}.nam",
        geometry.grid_corners_lonlat,
        geometry.lonlat_epsg,
    )
    LOG.info("Saved MODFLOW input files to %s", workdir)
    return SimulationBuildResult(sim=sim, gwf=gwf, material_fields=fields)
