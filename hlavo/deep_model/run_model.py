from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import struct
import tempfile
from pathlib import Path

import attrs
import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_hlavo"))

import flopy

LOG = logging.getLogger(__name__)




@attrs.define(frozen=True)
class RunConfig:
    config_path: Path
    workspace: Path
    model_name: str
    sim_name: str
    exe_name: str
    recharge_rate: float
    recharge_rate_override: float | None
    recharge_series_m_per_day: tuple[float, ...] | None
    drain_conductance: float
    use_uzf: bool
    use_drn: bool
    drain_mode: str
    initial_head_offset_override: float | None
    simulation_days: float
    stress_periods_days: tuple[float, ...]
    grid_output_path: Path
    material_parameters_path: Path
    paraview_output_dir: Path
    paraview_output_path: Path
    paraview_materials_output_path: Path
    paraview_quantities: tuple[str, ...]
    paraview_include_inactive: bool
    paraview_z_scale: float
    paraview_surface_timeseries: bool
    paraview_surface_timeseries_name: str
    plot_enabled: bool
    plot_output_dir: Path
    plot_dpi: int
    plot_active_mask_name: str
    plot_idomain_name: str
    plot_head_name: str
    plot_groundwater_surface_name: str
    plot_groundwater_change_name: str
    plot_hydrograph_name: str
    plot_xsection_x_times_name: str
    plot_xsection_y_times_name: str
    plot_velocity_name: str
    plot_xsection_x_name: str
    plot_xsection_y_name: str
    plot_xsection_y_index: int | None
    plot_xsection_x_index: int | None
    plot_xsection_depth_window: float
    plot_xsection_points: tuple[tuple[int, int], ...]
    plot_xsection_wells: tuple[str, ...]
    wells: tuple["WellSpec", ...]

    @staticmethod
    def from_yaml(config_path: Path, workspace: Path | None = None) -> "RunConfig":
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict), "Config YAML must be a mapping"

        model_raw = raw.get("model", {})
        assert isinstance(model_raw, dict), "model config must be a mapping"

        model_name = model_raw.get("model_name")
        assert model_name, "model.model_name is required in the config"
        model_name = str(model_name)

        sim_name = str(model_raw.get("sim_name", "uhelna"))
        exe_name = str(model_raw.get("exe_name", "mf6"))
        recharge_rate_override = None
        if "recharge_rate" in model_raw:
            recharge_rate_override = float(model_raw["recharge_rate"])
        recharge_rate = float(model_raw.get("recharge_rate", 1e-4))
        recharge_series_raw = model_raw.get("recharge_series_m_per_day")
        if recharge_series_raw is not None:
            assert isinstance(recharge_series_raw, (list, tuple)), (
                "model.recharge_series_m_per_day must be a list"
            )
            recharge_series_m_per_day = tuple(float(value) for value in recharge_series_raw)
            assert all(np.isfinite(value) for value in recharge_series_m_per_day), (
                "model.recharge_series_m_per_day must contain finite values"
            )
        else:
            recharge_series_m_per_day = None
        drain_conductance = float(model_raw.get("drain_conductance", 1.0))
        use_uzf = bool(model_raw.get("use_uzf", False))
        use_drn_raw = model_raw.get("use_drn")
        if use_drn_raw is None:
            use_drn = not use_uzf
        else:
            use_drn = bool(use_drn_raw)
        drain_mode = str(model_raw.get("drain_mode", "top"))
        assert drain_mode in ("top", "north_side"), (
            "model.drain_mode must be one of: top, north_side"
        )
        initial_head_offset_override = None
        if "initial_head_offset" in model_raw:
            initial_head_offset_override = float(model_raw["initial_head_offset"])
        simulation_days = model_raw.get("simulation_days")
        if simulation_days is not None:
            simulation_days = float(simulation_days)
            assert simulation_days > 0.0, "model.simulation_days must be > 0"

        stress_periods_raw = model_raw.get("stress_periods_days")
        if stress_periods_raw is not None:
            assert isinstance(
                stress_periods_raw, (list, tuple)
            ), "model.stress_periods_days must be a list"
            stress_periods_days = tuple(float(value) for value in stress_periods_raw)
            assert all(value > 0.0 for value in stress_periods_days), (
                "model.stress_periods_days values must be > 0"
            )
        else:
            stress_periods_days = ()

        if simulation_days is None and not stress_periods_days:
            simulation_days = 1.0
            stress_periods_days = (simulation_days,)
        elif simulation_days is None:
            simulation_days = float(sum(stress_periods_days))
        elif not stress_periods_days:
            stress_periods_days = (simulation_days,)
        else:
            total = float(sum(stress_periods_days))
            assert np.isclose(total, simulation_days, rtol=1e-6, atol=1e-6), (
                "Sum of model.stress_periods_days must equal model.simulation_days"
            )
        if recharge_series_m_per_day is not None:
            assert len(recharge_series_m_per_day) == len(stress_periods_days), (
                "model.recharge_series_m_per_day length must match stress periods"
            )

        base_workspace = Path(workspace) if workspace is not None else Path(
            model_raw.get("workspace", "model")
        )
        run_workspace = base_workspace / model_name

        output_grid_raw = raw.get("grid_output_path")
        if output_grid_raw:
            grid_output_path = Path(str(output_grid_raw))
            if not grid_output_path.is_absolute():
                grid_output_path = run_workspace / grid_output_path
        else:
            grid_output_path = run_workspace / "grid_materials.npz"

        material_parameters_raw = raw.get("material_parameters_output_path")
        if material_parameters_raw:
            material_parameters_path = Path(str(material_parameters_raw))
            if not material_parameters_path.is_absolute():
                material_parameters_path = run_workspace / material_parameters_path
        else:
            material_parameters_path = run_workspace / "material_parameters.npz"

        paraview_cfg = raw.get("paraview", model_raw.get("paraview", {}))
        assert isinstance(paraview_cfg, dict), "paraview config must be a mapping"

        paraview_output_dir_raw = paraview_cfg.get("output_dir", "paraview")
        paraview_output_dir = Path(str(paraview_output_dir_raw))
        if not paraview_output_dir.is_absolute():
            paraview_output_dir = run_workspace / paraview_output_dir

        legacy_results_raw = model_raw.get("paraview_results_output", raw.get("paraview_results_output"))
        if legacy_results_raw:
            paraview_output_path = Path(str(legacy_results_raw))
            if not paraview_output_path.is_absolute():
                paraview_output_path = run_workspace / paraview_output_path
        else:
            paraview_results_name = str(paraview_cfg.get("results_name", f"{sim_name}_results.vtu"))
            paraview_output_path = paraview_output_dir / paraview_results_name

        legacy_materials_raw = model_raw.get("paraview_grid_output", raw.get("paraview_grid_output"))
        if legacy_materials_raw:
            paraview_materials_output_path = Path(str(legacy_materials_raw))
            if not paraview_materials_output_path.is_absolute():
                paraview_materials_output_path = run_workspace / paraview_materials_output_path
        else:
            paraview_materials_name = str(paraview_cfg.get("materials_name", f"{sim_name}_materials.vtr"))
            paraview_materials_output_path = paraview_output_dir / paraview_materials_name

        quantities_raw = paraview_cfg.get("quantities")
        if quantities_raw is None:
            paraview_quantities = (
                "head",
                "hk",
                "idomain",
                "materials",
                "groundwater_surface_change",
                "velocity",
                "q",
                "velocity_magnitude",
                "active",
            )
        else:
            assert isinstance(quantities_raw, (list, tuple)), "paraview.quantities must be a list"
            paraview_quantities = tuple(str(item) for item in quantities_raw)
        paraview_include_inactive = bool(paraview_cfg.get("include_inactive", False))
        paraview_z_scale = float(paraview_cfg.get("z_scale", 1.0))
        assert paraview_z_scale > 0.0, "paraview.z_scale must be > 0"
        paraview_surface_timeseries = bool(paraview_cfg.get("groundwater_surface_timeseries", False))
        paraview_surface_timeseries_name = str(
            paraview_cfg.get("groundwater_surface_timeseries_name", f"{sim_name}_gw_surface_timeseries.pvd")
        )

        plot_raw = raw.get("plots", {})
        assert isinstance(plot_raw, dict), "plots config must be a mapping"
        plot_enabled = bool(plot_raw.get("enabled", True))
        plot_output_raw = plot_raw.get("output_dir")
        if plot_output_raw:
            plot_output_dir = Path(str(plot_output_raw))
            if not plot_output_dir.is_absolute():
                plot_output_dir = run_workspace / plot_output_dir
        else:
            plot_output_dir = run_workspace / "plots"
        plot_dpi = int(plot_raw.get("dpi", 150))
        plot_active_mask_name = str(plot_raw.get("active_mask_name", "grid_active_mask.png"))
        plot_idomain_name = str(plot_raw.get("idomain_name", "idomain_top.png"))
        plot_head_name = str(plot_raw.get("head_name", "head_groundplan.png"))
        plot_groundwater_surface_name = str(
            plot_raw.get("groundwater_surface_name", "groundwater_surface.png")
        )
        plot_groundwater_change_name = str(
            plot_raw.get("groundwater_change_name", "groundwater_change.png")
        )
        plot_hydrograph_name = str(plot_raw.get("hydrograph_name", "groundwater_hydrograph.png"))
        plot_xsection_x_times_name = str(
            plot_raw.get("xsection_x_times_name", "groundwater_x_section_times.png")
        )
        plot_xsection_y_times_name = str(
            plot_raw.get("xsection_y_times_name", "groundwater_y_section_times.png")
        )
        plot_velocity_name = str(plot_raw.get("velocity_name", "velocity_groundplan.png"))
        plot_xsection_x_name = str(plot_raw.get("xsection_x_name", "materials_x_section.png"))
        plot_xsection_y_name = str(plot_raw.get("xsection_y_name", "materials_y_section.png"))
        plot_xsection_y_index = plot_raw.get("xsection_y_index")
        if plot_xsection_y_index is not None:
            plot_xsection_y_index = int(plot_xsection_y_index)
        plot_xsection_x_index = plot_raw.get("xsection_x_index")
        if plot_xsection_x_index is not None:
            plot_xsection_x_index = int(plot_xsection_x_index)
        plot_xsection_depth_window = float(plot_raw.get("xsection_depth_window", 40.0))
        assert plot_xsection_depth_window > 0.0, "plots.xsection_depth_window must be > 0"

        wells = _parse_wells(raw)
        plot_xsection_points = _parse_plot_xsection_points(plot_raw)
        plot_xsection_wells = _parse_plot_xsection_wells(plot_raw)

        return RunConfig(
            config_path=config_path,
            workspace=run_workspace,
            model_name=model_name,
            sim_name=sim_name,
            exe_name=exe_name,
            recharge_rate=recharge_rate,
            recharge_rate_override=recharge_rate_override,
            recharge_series_m_per_day=recharge_series_m_per_day,
            drain_conductance=drain_conductance,
            use_uzf=use_uzf,
            use_drn=use_drn,
            drain_mode=drain_mode,
            initial_head_offset_override=initial_head_offset_override,
            simulation_days=simulation_days,
            stress_periods_days=stress_periods_days,
            grid_output_path=grid_output_path,
            material_parameters_path=material_parameters_path,
            paraview_output_dir=paraview_output_dir,
            paraview_output_path=paraview_output_path,
            paraview_materials_output_path=paraview_materials_output_path,
            paraview_quantities=paraview_quantities,
            paraview_include_inactive=paraview_include_inactive,
            paraview_z_scale=paraview_z_scale,
            paraview_surface_timeseries=paraview_surface_timeseries,
            paraview_surface_timeseries_name=paraview_surface_timeseries_name,
            plot_enabled=plot_enabled,
            plot_output_dir=plot_output_dir,
            plot_dpi=plot_dpi,
            plot_active_mask_name=plot_active_mask_name,
            plot_idomain_name=plot_idomain_name,
            plot_head_name=plot_head_name,
            plot_groundwater_surface_name=plot_groundwater_surface_name,
            plot_groundwater_change_name=plot_groundwater_change_name,
            plot_hydrograph_name=plot_hydrograph_name,
            plot_xsection_x_times_name=plot_xsection_x_times_name,
            plot_xsection_y_times_name=plot_xsection_y_times_name,
            plot_velocity_name=plot_velocity_name,
            plot_xsection_x_name=plot_xsection_x_name,
            plot_xsection_y_name=plot_xsection_y_name,
            plot_xsection_y_index=plot_xsection_y_index,
            plot_xsection_x_index=plot_xsection_x_index,
            plot_xsection_depth_window=plot_xsection_depth_window,
            plot_xsection_points=plot_xsection_points,
            plot_xsection_wells=plot_xsection_wells,
            wells=wells,
        )

@attrs.define(frozen=True)
class WellInterval:
    top_depth: float
    bottom_depth: float
    rate_m3_per_day: float


@attrs.define(frozen=True)
class WellSpec:
    name: str
    lon: float
    lat: float
    intervals: tuple[WellInterval, ...]


def _parse_wells(raw: dict) -> tuple[WellSpec, ...]:
    wells_raw = raw.get("wells")
    if wells_raw is None:
        return ()
    assert isinstance(wells_raw, list), "wells must be a list"
    wells: list[WellSpec] = []
    for idx, well_raw in enumerate(wells_raw):
        assert isinstance(well_raw, dict), f"wells[{idx}] must be a mapping"
        name = str(well_raw.get("name") or f"well_{idx + 1}")

        location_raw = well_raw.get("location")
        assert isinstance(location_raw, dict), f"wells[{idx}].location must be a mapping"
        location_type = str(location_raw.get("type", "direct"))
        assert location_type == "direct", (
            "Only wells.location.type=direct is supported at the moment"
        )
        lon = location_raw.get("lon")
        lat = location_raw.get("lat")
        assert lon is not None and lat is not None, (
            f"wells[{idx}].location.lon and wells[{idx}].location.lat are required"
        )
        lon = float(lon)
        lat = float(lat)
        assert np.isfinite(lon) and np.isfinite(lat), f"wells[{idx}] lon/lat must be finite"

        intervals_raw = well_raw.get("intervals")
        if intervals_raw is None and "interval" in well_raw:
            intervals_raw = [well_raw["interval"]]
        assert intervals_raw is not None, (
            f"wells[{idx}] must define interval or intervals"
        )
        assert isinstance(intervals_raw, (list, tuple)), (
            f"wells[{idx}].intervals must be a list"
        )
        intervals: list[WellInterval] = []
        for jdx, interval_raw in enumerate(intervals_raw):
            assert isinstance(interval_raw, dict), (
                f"wells[{idx}].intervals[{jdx}] must be a mapping"
            )
            top_depth = interval_raw.get("top_depth")
            bottom_depth = interval_raw.get("bottom_depth")
            assert top_depth is not None and bottom_depth is not None, (
                f"wells[{idx}].intervals[{jdx}] must define top_depth and bottom_depth"
            )
            top_depth = float(top_depth)
            bottom_depth = float(bottom_depth)
            assert top_depth >= 0.0, (
                f"wells[{idx}].intervals[{jdx}].top_depth must be >= 0"
            )
            assert bottom_depth > top_depth, (
                f"wells[{idx}].intervals[{jdx}].bottom_depth must be > top_depth"
            )

            rate = interval_raw.get("rate_m3_per_day", interval_raw.get("rate"))
            assert rate is not None, (
                f"wells[{idx}].intervals[{jdx}] must define rate_m3_per_day"
            )
            rate = float(rate)
            assert np.isfinite(rate), (
                f"wells[{idx}].intervals[{jdx}].rate_m3_per_day must be finite"
            )
            intervals.append(
                WellInterval(
                    top_depth=top_depth,
                    bottom_depth=bottom_depth,
                    rate_m3_per_day=rate,
                )
            )
        assert intervals, f"wells[{idx}] must contain at least one interval"
        wells.append(
            WellSpec(
                name=name,
                lon=lon,
                lat=lat,
                intervals=tuple(intervals),
            )
        )
    return tuple(wells)


def _parse_plot_xsection_points(plot_raw: dict) -> tuple[tuple[int, int], ...]:
    points_raw = plot_raw.get("xsection_points")
    if points_raw is None:
        return ()
    assert isinstance(points_raw, list), "plots.xsection_points must be a list"
    points: list[tuple[int, int]] = []
    for idx, point_raw in enumerate(points_raw):
        assert isinstance(point_raw, dict), f"plots.xsection_points[{idx}] must be a mapping"
        x_index = point_raw.get("x_index")
        y_index = point_raw.get("y_index")
        assert x_index is not None and y_index is not None, (
            f"plots.xsection_points[{idx}] must define x_index and y_index"
        )
        x_index = int(x_index)
        y_index = int(y_index)
        points.append((x_index, y_index))
    return tuple(points)


def _parse_plot_xsection_wells(plot_raw: dict) -> tuple[str, ...]:
    wells_raw = plot_raw.get("xsection_wells")
    if wells_raw is None:
        return ()
    assert isinstance(wells_raw, (list, tuple)), "plots.xsection_wells must be a list"
    return tuple(str(value) for value in wells_raw)


def _grid_arrays_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    assert npz_path.exists(), f"Grid NPZ not found: {npz_path}"
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scalar_from_model_data(
    model_data: dict[str, np.ndarray],
    key: str,
    *,
    cast=float,
) -> float | int | bool:
    assert key in model_data, f"Missing '{key}' in material parameters"
    value = np.asarray(model_data[key]).reshape(-1)[0]
    if cast is bool:
        return bool(float(value))
    return cast(value)


def _layer_tops_from_top_botm(top: np.ndarray, botm: np.ndarray) -> np.ndarray:
    assert top.ndim == 2, "top must be 2D"
    assert botm.ndim == 3, "botm must be 3D"
    nz = botm.shape[0]
    layer_tops = np.empty_like(botm)
    layer_tops[0] = top
    if nz > 1:
        layer_tops[1:] = botm[:-1]
    return layer_tops


def _lonlat_to_local_xy(
    lon: float,
    lat: float,
    boundary_origin: np.ndarray,
) -> tuple[float, float]:
    from pyproj import Transformer

    transformer = Transformer.from_crs(4326, 5514, always_xy=True)
    x_global, y_global = transformer.transform(lon, lat)
    origin = np.asarray(boundary_origin, dtype=float)
    assert origin.shape[0] >= 2, "boundary_origin must have at least 2 values"
    x_local = float(x_global - origin[0])
    y_local = float(y_global - origin[1])
    return x_local, y_local


def _row_col_from_xy(
    x_local: float,
    y_local: float,
    origin: np.ndarray,
    step: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    origin = np.asarray(origin, dtype=float)
    step = np.asarray(step, dtype=float)
    assert origin.shape[0] >= 2, "origin must have at least 2 values"
    assert step.shape[0] >= 2, "step must have at least 2 values"
    col = int(np.floor((x_local - origin[0]) / step[0]))
    row = int(np.floor((y_local - origin[1]) / step[1]))
    assert 0 <= col < nx and 0 <= row < ny, (
        f"Well location outside grid: row={row}, col={col}, grid={ny}x{nx}"
    )
    return row, col


def _well_cell_rates(
    *,
    row: int,
    col: int,
    interval: WellInterval,
    top: np.ndarray,
    botm: np.ndarray,
    layer_tops: np.ndarray,
    idomain: np.ndarray,
) -> list[tuple[int, int, int, float]]:
    nz = botm.shape[0]
    surface = float(top[row, col])
    z_top = surface - interval.top_depth
    z_bottom = surface - interval.bottom_depth
    overlaps: list[tuple[int, float]] = []
    for lay in range(nz):
        if idomain[lay, row, col] <= 0:
            continue
        cell_top = float(layer_tops[lay, row, col])
        cell_bot = float(botm[lay, row, col])
        overlap = min(cell_top, z_top) - max(cell_bot, z_bottom)
        if overlap > 0.0:
            overlaps.append((lay, overlap))
    assert overlaps, (
        f"Well interval does not intersect active cells at row={row}, col={col}"
    )
    total = float(sum(thick for _, thick in overlaps))
    assert total > 0.0, "Well interval overlap thickness must be > 0"

    cells: list[tuple[int, int, int, float]] = []
    for lay, overlap in overlaps:
        rate = interval.rate_m3_per_day * (overlap / total)
        cells.append((int(lay), int(row), int(col), float(rate)))
    return cells


def _build_well_spd(
    wells: tuple[WellSpec, ...],
    grid_data: dict[str, np.ndarray],
    top: np.ndarray,
    botm: np.ndarray,
    idomain: np.ndarray,
    nper: int,
) -> dict[int, list[tuple[int, int, int, float]]]:
    if not wells:
        return {}

    el_dims = np.asarray(grid_data["el_dims"], dtype=int)
    nx = int(el_dims[0])
    ny = int(el_dims[1])
    nz = int(el_dims[2])
    assert botm.shape[0] == nz, "botm nz mismatch"
    origin = np.asarray(grid_data["origin"], dtype=float)
    step = np.asarray(grid_data["step"], dtype=float)
    boundary_origin = np.asarray(grid_data["boundary_origin"], dtype=float)
    active_mask = np.asarray(grid_data["active_mask"], dtype=bool)

    layer_tops = _layer_tops_from_top_botm(top, botm)
    aggregated: dict[tuple[int, int, int], float] = {}

    for well in wells:
        x_local, y_local = _lonlat_to_local_xy(well.lon, well.lat, boundary_origin)
        row, col = _row_col_from_xy(x_local, y_local, origin, step, nx, ny)
        assert active_mask[row, col], (
            f"Well {well.name} is outside active model domain at row={row}, col={col}"
        )
        if not np.any(idomain[:, row, col] > 0):
            raise AssertionError(
                f"Well {well.name} is in an inactive column at row={row}, col={col}"
            )
        for interval in well.intervals:
            cells = _well_cell_rates(
                row=row,
                col=col,
                interval=interval,
                top=top,
                botm=botm,
                layer_tops=layer_tops,
                idomain=idomain,
            )
            for lay, r, c, rate in cells:
                key = (lay, r, c)
                aggregated[key] = aggregated.get(key, 0.0) + float(rate)
        LOG.info(
            "Well %s mapped to row=%s col=%s with %s interval(s)",
            well.name,
            row,
            col,
            len(well.intervals),
        )

    stress_period_data = [
        (lay, row, col, rate) for (lay, row, col), rate in aggregated.items()
    ]
    assert stress_period_data, "No well cells mapped to the grid"
    return {iper: stress_period_data for iper in range(nper)}


def _vtk_cell_array(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 3, "Cell array must be 3D"
    cell_values = np.transpose(values, (2, 1, 0))
    return cell_values.ravel(order="F")


def _vtk_oriented(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 3, "Values must be 3D"
    # Flip Z and Y to align array indexing (top row first, top layer last) with
    # increasing coordinate axes used by VTK.
    return values[::-1, ::-1, :]


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


def _surface_to_cell_values(surface: np.ndarray, idomain: np.ndarray) -> np.ndarray:
    assert surface.ndim == 2, "surface must be 2D"
    assert idomain.ndim == 3, "idomain must be 3D"
    nz, ny, nx = idomain.shape
    assert surface.shape == (ny, nx), "surface shape mismatch"
    values = np.full((nz, ny, nx), np.nan, dtype=float)
    active_top = idomain[0] > 0
    values[0] = np.where(active_top, surface, np.nan)
    return values


def _material_class_from_model_data(
    model_data: dict[str, np.ndarray],
    materials: np.ndarray,
) -> np.ndarray:
    assert materials.ndim == 3, "materials must be 3D"
    if "material_class" in model_data:
        classes = np.asarray(model_data["material_class"], dtype=np.int16)
        assert classes.shape == materials.shape, "material_class shape mismatch"
        return classes

    classes = np.full(materials.shape, -1, dtype=np.int16)  # -1=inactive/undefined
    layer_names = [str(name) for name in model_data.get("layer_names", np.asarray([], dtype=object)).tolist()]
    if not layer_names:
        classes[materials >= 0] = 0
        return classes

    class_by_layer = np.zeros(len(layer_names), dtype=np.int16)  # 0=other, 1=sand, 2=clay
    for idx, layer_name in enumerate(layer_names):
        if layer_name.startswith("Q") and layer_name.endswith("_base"):
            class_by_layer[idx] = 1
        elif layer_name.startswith("Q") and layer_name.endswith("_top"):
            class_by_layer[idx] = 2
    valid = materials >= 0
    classes[valid] = class_by_layer[materials[valid]]
    return classes


def _surface_to_nodes(surface: np.ndarray) -> np.ndarray:
    assert surface.ndim == 2, "Surface must be 2D"
    ny, nx = surface.shape
    nodes = np.empty((ny + 1, nx + 1), dtype=float)
    nodes[1:-1, 1:-1] = 0.25 * (
        surface[:-1, :-1]
        + surface[:-1, 1:]
        + surface[1:, :-1]
        + surface[1:, 1:]
    )
    nodes[0, 1:-1] = 0.5 * (surface[0, :-1] + surface[0, 1:])
    nodes[-1, 1:-1] = 0.5 * (surface[-1, :-1] + surface[-1, 1:])
    nodes[1:-1, 0] = 0.5 * (surface[:-1, 0] + surface[1:, 0])
    nodes[1:-1, -1] = 0.5 * (surface[:-1, -1] + surface[1:, -1])
    nodes[0, 0] = surface[0, 0]
    nodes[0, -1] = surface[0, -1]
    nodes[-1, 0] = surface[-1, 0]
    nodes[-1, -1] = surface[-1, -1]
    return nodes


def _deformed_structured_grid(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
    z_scale: float,
) -> "pv.StructuredGrid":
    import pyvista as pv

    assert top.ndim == 2, "Top surface must be 2D"
    assert botm.ndim == 3, "Botm must be 3D"
    assert z_scale > 0.0, "z_scale must be > 0"
    nz, ny, nx = botm.shape
    assert top.shape == (ny, nx), "Top shape mismatch with botm"

    # Build surfaces from bottom to top in VTK coordinate order.
    surfaces = [botm[k] for k in range(nz - 1, -1, -1)] + [top]
    # Orient surfaces so Y increases with coordinate axes.
    node_surfaces = [_surface_to_nodes(surface[::-1, :]) for surface in surfaces]

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    x_3d = np.repeat(x_grid[:, :, None], nz + 1, axis=2)
    y_3d = np.repeat(y_grid[:, :, None], nz + 1, axis=2)
    z_3d = np.empty((x_nodes.size, y_nodes.size, nz + 1), dtype=float)
    for k, nodes in enumerate(node_surfaces):
        z_3d[:, :, k] = nodes.T
    if z_scale != 1.0:
        z_ref = float(np.nanmax(z_3d))
        z_3d = z_ref + (z_3d - z_ref) * z_scale

    return pv.StructuredGrid(x_3d, y_3d, z_3d)


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
    material_class: np.ndarray | None,
    top: np.ndarray,
    botm: np.ndarray,
    paraview_quantities: tuple[str, ...],
    include_inactive: bool,
    z_scale: float,
    groundwater_surface: np.ndarray | None = None,
    groundwater_surface_change: np.ndarray | None = None,
) -> Path:
    import pyvista as pv
    from pyvista.core.pointset import UnstructuredGrid

    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = _deformed_structured_grid(x_nodes, y_nodes, top, botm, z_scale)

    quantities = set(paraview_quantities)
    head_masked = np.where(idomain > 0, head, np.nan)
    qx_masked = np.where(idomain > 0, qx, np.nan)
    qy_masked = np.where(idomain > 0, qy, np.nan)
    qz_masked = np.where(idomain > 0, qz, np.nan)
    idomain_vtk = _vtk_cell_array(_vtk_oriented(idomain).astype(np.int8))

    if "head" in quantities:
        grid.cell_data["head"] = _vtk_cell_array(_vtk_oriented(head_masked))
    if "hk" in quantities:
        grid.cell_data["hk"] = _vtk_cell_array(_vtk_oriented(hk))
    if "idomain" in quantities:
        grid.cell_data["idomain"] = idomain_vtk
    if "materials" in quantities:
        grid.cell_data["materials"] = _vtk_cell_array(_vtk_oriented(materials.astype(np.int16)))
    if material_class is not None:
        grid.cell_data["material_class"] = _vtk_cell_array(
            _vtk_oriented(material_class.astype(np.int16))
        )
    if "groundwater_surface" in quantities and groundwater_surface is not None:
        gw_values = _surface_to_cell_values(groundwater_surface, idomain)
        grid.cell_data["groundwater_surface"] = _vtk_cell_array(_vtk_oriented(gw_values))
    if "groundwater_surface_change" in quantities and groundwater_surface_change is not None:
        gw_change = _surface_to_cell_values(groundwater_surface_change, idomain)
        grid.cell_data["groundwater_surface_change"] = _vtk_cell_array(_vtk_oriented(gw_change))

    needs_vectors = "velocity" in quantities or "q" in quantities
    needs_qmag = "velocity_magnitude" in quantities
    if needs_vectors or needs_qmag:
        vectors = _vtk_cell_vectors(
            _vtk_oriented(qx_masked),
            _vtk_oriented(qy_masked),
            _vtk_oriented(qz_masked),
        )
        if "velocity" in quantities:
            grid.cell_data["velocity"] = vectors
        if "q" in quantities:
            grid.cell_data["q"] = vectors
        if needs_qmag:
            qmag = np.sqrt(qx_masked**2 + qy_masked**2 + qz_masked**2)
            grid.cell_data["velocity_magnitude"] = _vtk_cell_array(_vtk_oriented(qmag))

    active_grid = grid
    if not include_inactive:
        active_cells = np.flatnonzero(idomain_vtk > 0)
        active_grid = grid.extract_cells(active_cells)
    output_path = Path(output_path)
    structured_ext = {".vtk", ".vts", ".pkl", ".pickle"}
    unstructured_ext = {".vtu", ".vtk", ".vtkhdf", ".pkl", ".pickle"}
    if isinstance(active_grid, UnstructuredGrid):
        if output_path.suffix.lower() not in unstructured_ext:
            corrected = output_path.with_suffix(".vtu")
            LOG.warning(
                "Paraview output extension %s invalid for UnstructuredGrid; using %s instead.",
                output_path.suffix,
                corrected,
            )
            output_path = corrected
    elif output_path.suffix.lower() not in structured_ext:
        corrected = output_path.with_suffix(".vts")
        LOG.warning(
            "Paraview output extension %s invalid for StructuredGrid; using %s instead.",
            output_path.suffix,
            corrected,
        )
        output_path = corrected
    active_grid.save(str(output_path))
    assert output_path.exists(), f"Failed to write Paraview results: {output_path}"
    LOG.info("Saved Paraview results to %s", output_path)
    return output_path


def _export_materials_to_paraview(
    output_path: Path,
    grid_data: dict[str, np.ndarray],
    materials: np.ndarray,
    material_class: np.ndarray | None,
    active_mask: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
    paraview_quantities: tuple[str, ...],
    include_inactive: bool,
    z_scale: float,
) -> Path:
    import pyvista as pv
    from pyvista.core.pointset import UnstructuredGrid

    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)

    assert materials.ndim == 3, "Materials array must be 3D"
    assert active_mask.ndim == 2, "Active mask must be 2D"

    active_cells = np.broadcast_to(active_mask, materials.shape)

    grid = _deformed_structured_grid(x_nodes, y_nodes, top, botm, z_scale)
    quantities = set(paraview_quantities)
    active_vtk = _vtk_cell_array(_vtk_oriented(active_cells.astype(np.uint8)))
    if "materials" in quantities:
        grid.cell_data["materials"] = _vtk_cell_array(_vtk_oriented(materials.astype(np.int16)))
    if material_class is not None:
        grid.cell_data["material_class"] = _vtk_cell_array(
            _vtk_oriented(material_class.astype(np.int16))
        )
    if "active" in quantities:
        grid.cell_data["active"] = active_vtk

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_grid = grid
    if not include_inactive:
        active_ids = np.flatnonzero(active_vtk > 0)
        active_grid = grid.extract_cells(active_ids)
    structured_ext = {".vtk", ".vts", ".pkl", ".pickle"}
    unstructured_ext = {".vtu", ".vtk", ".vtkhdf", ".pkl", ".pickle"}
    if isinstance(active_grid, UnstructuredGrid):
        if output_path.suffix.lower() not in unstructured_ext:
            corrected = output_path.with_suffix(".vtu")
            LOG.warning(
                "Paraview materials extension %s invalid for UnstructuredGrid; using %s instead.",
                output_path.suffix,
                corrected,
            )
            output_path = corrected
    elif output_path.suffix.lower() not in structured_ext:
        corrected = output_path.with_suffix(".vts")
        LOG.warning(
            "Paraview materials extension %s invalid for StructuredGrid; using %s instead.",
            output_path.suffix,
            corrected,
        )
        output_path = corrected
    active_grid.save(str(output_path))
    assert output_path.exists(), f"Failed to write Paraview materials: {output_path}"
    LOG.info("Saved Paraview materials to %s", output_path)
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
    botm: np.ndarray,
    material_labels: list[str] | None = None,
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
    if not np.isfinite(head_plan).any():
        groundwater_surface = _groundwater_surface_from_head(head, idomain, top, botm)
        head_plan = groundwater_surface
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

    groundwater_surface = _groundwater_surface_from_head(head, idomain, top, botm)
    plt.figure(figsize=(8, 6))
    img = plt.imshow(groundwater_surface, origin="upper", extent=extent, cmap="cividis")
    plt.title("Groundwater Surface Elevation (Plan View)")
    plt.xlabel("X (local)")
    plt.ylabel("Y (local)")
    plt.colorbar(img, label="Groundwater surface Z")
    plt.tight_layout()
    gw_surface_path = plot_dir / run_config.plot_groundwater_surface_name
    plt.savefig(gw_surface_path, dpi=run_config.plot_dpi)
    plt.close()
    LOG.info("Saved groundwater surface plot to %s", gw_surface_path)

    velocity_plan = np.where(idomain[0] > 0, np.sqrt(qx[0] ** 2 + qy[0] ** 2), np.nan)
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_centers = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    xv, yv = np.meshgrid(x_centers, y_centers)
    stride = max(1, int(max(x_centers.size, y_centers.size) / 30))
    xs = xv[::stride, ::stride]
    ys = yv[::stride, ::stride]
    u = qx[0][::stride, ::stride]
    v = qy[0][::stride, ::stride]
    mask = ~np.isfinite(u) | ~np.isfinite(v) | (idomain[0][::stride, ::stride] <= 0)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(velocity_plan, origin="upper", extent=extent, cmap="magma")
    ax.quiver(xs, ys, u, v, angles="xy", scale_units="xy", scale=None, width=0.002)
    ax.set_title("Flow Velocity Vectors (Top Layer, Plan View)")
    ax.set_xlabel("X (local)")
    ax.set_ylabel("Y (local)")
    ax.set_ylim(y_nodes[-1], y_nodes[0])
    fig.colorbar(img, ax=ax, label="Velocity magnitude")
    fig.tight_layout()
    velocity_path = plot_dir / run_config.plot_velocity_name
    fig.savefig(velocity_path, dpi=run_config.plot_dpi)
    plt.close(fig)
    LOG.info("Saved velocity plot to %s", velocity_path)

    _write_cross_section_plots(
        run_config,
        grid_data,
        materials,
        top,
        botm,
        groundwater_surface,
        material_labels,
    )


def _groundwater_surface_from_head(
    head: np.ndarray,
    idomain: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
) -> np.ndarray:
    assert head.ndim == 3 and idomain.ndim == 3 and botm.ndim == 3
    nz, ny, nx = head.shape
    assert idomain.shape == (nz, ny, nx), "idomain shape mismatch"
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"
    assert top.shape == (ny, nx), "top shape mismatch"

    head_clean = np.asarray(head, dtype=float)
    head_clean = np.where(np.isfinite(head_clean), head_clean, np.nan)
    head_clean = np.where(np.abs(head_clean) > 1.0e20, np.nan, head_clean)

    water_table = np.full((ny, nx), np.nan, dtype=float)
    for k in range(nz):
        layer_active = idomain[k] > 0
        layer_top = top if k == 0 else botm[k - 1]
        layer_bot = botm[k]
        layer_head = head_clean[k]
        valid = layer_active & np.isfinite(layer_head)
        saturated = valid & (layer_head > layer_bot)
        assign = np.isnan(water_table) & (saturated | valid)
        if np.any(assign):
            clipped = np.where(layer_head > layer_bot, np.minimum(layer_head, layer_top), layer_head)
            water_table[assign] = clipped[assign]
    return water_table


def _write_cross_section_plots(
    run_config: RunConfig,
    grid_data: dict[str, np.ndarray],
    materials: np.ndarray,
    top: np.ndarray,
    botm: np.ndarray,
    groundwater_surface: np.ndarray,
    material_labels: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    x_nodes = grid_data["x_nodes"].astype(float)
    y_nodes = grid_data["y_nodes"].astype(float)
    nz, ny, nx = materials.shape
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"
    if material_labels is None:
        layer_names = [str(name) for name in grid_data["layer_names"].tolist()]
    else:
        layer_names = [str(name) for name in material_labels]
    assert len(layer_names) > 0, "layer_names cannot be empty"
    assert len(layer_names) <= 256, "Too many layer names for discrete colormap"
    cmap = plt.get_cmap("tab20", len(layer_names))
    norm = colors.BoundaryNorm(np.arange(-0.5, len(layer_names) + 0.5, 1.0), cmap.N)

    y_index = run_config.plot_xsection_y_index
    if y_index is None:
        y_index = ny // 2
    assert 0 <= y_index < ny, "xsection_y_index out of range"

    x_index = run_config.plot_xsection_x_index
    if x_index is None:
        x_index = nx // 2
    assert 0 <= x_index < nx, "xsection_x_index out of range"

    # X-direction cross-section at fixed Y index.
    materials_x = materials[:, y_index, :]
    top_line_x = top[y_index, :]
    botm_x = botm[:, y_index, :]
    gw_line_x = groundwater_surface[y_index, :]
    z_edges_center = np.empty((nz + 1, nx), dtype=float)
    z_edges_center[0] = top_line_x
    z_edges_center[1:] = botm_x
    z_edges = np.empty((nz + 1, nx + 1), dtype=float)
    z_edges[:, 1:-1] = 0.5 * (z_edges_center[:, :-1] + z_edges_center[:, 1:])
    z_edges[:, 0] = z_edges_center[:, 0]
    z_edges[:, -1] = z_edges_center[:, -1]
    x_edges = np.tile(x_nodes[None, :], (nz + 1, 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    mesh = ax.pcolormesh(
        x_edges,
        z_edges,
        np.ma.masked_less(materials_x, 0),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    ax.plot(x_centers, gw_line_x, "b-", linewidth=1.5, label="groundwater surface")
    z_top_x = float(np.nanmax(top_line_x))
    z_bottom_x = float(np.nanmin(top_line_x - run_config.plot_xsection_depth_window))
    z_pad_x = 0.05 * run_config.plot_xsection_depth_window
    ax.set_ylim(z_bottom_x, z_top_x + z_pad_x)
    ax.set_title(f"Materials X-Section (y index {y_index})")
    ax.set_xlabel("X (local)")
    ax.set_ylabel("Z")
    ax.legend(loc="best")
    ticks = np.arange(len(layer_names), dtype=float)
    cbar = fig.colorbar(mesh, ax=ax, ticks=ticks, pad=0.02)
    cbar.ax.set_yticklabels(layer_names)
    cbar.set_label("Geological layer")
    fig.tight_layout()
    xsec_path = Path(run_config.plot_output_dir) / run_config.plot_xsection_x_name
    fig.savefig(xsec_path, dpi=run_config.plot_dpi)
    plt.close(fig)
    LOG.info("Saved X-section plot to %s", xsec_path)

    # Y-direction cross-section at fixed X index.
    materials_y = materials[:, :, x_index]
    top_line_y = top[:, x_index]
    botm_y = botm[:, :, x_index]
    gw_line_y = groundwater_surface[:, x_index]
    z_edges_center = np.empty((nz + 1, ny), dtype=float)
    z_edges_center[0] = top_line_y
    z_edges_center[1:] = botm_y
    z_edges = np.empty((nz + 1, ny + 1), dtype=float)
    z_edges[:, 1:-1] = 0.5 * (z_edges_center[:, :-1] + z_edges_center[:, 1:])
    z_edges[:, 0] = z_edges_center[:, 0]
    z_edges[:, -1] = z_edges_center[:, -1]
    y_edges = np.tile(y_nodes[None, :], (nz + 1, 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    mesh = ax.pcolormesh(
        y_edges,
        z_edges,
        np.ma.masked_less(materials_y, 0),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    y_centers = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    ax.plot(y_centers, gw_line_y, "b-", linewidth=1.5, label="groundwater surface")
    z_top_y = float(np.nanmax(top_line_y))
    z_bottom_y = float(np.nanmin(top_line_y - run_config.plot_xsection_depth_window))
    z_pad_y = 0.05 * run_config.plot_xsection_depth_window
    ax.set_ylim(z_bottom_y, z_top_y + z_pad_y)
    ax.set_title(f"Materials Y-Section (x index {x_index})")
    ax.set_xlabel("Y (local)")
    ax.set_ylabel("Z")
    ax.legend(loc="best")
    ticks = np.arange(len(layer_names), dtype=float)
    cbar = fig.colorbar(mesh, ax=ax, ticks=ticks, pad=0.02)
    cbar.ax.set_yticklabels(layer_names)
    cbar.set_label("Geological layer")
    fig.tight_layout()
    ysec_path = Path(run_config.plot_output_dir) / run_config.plot_xsection_y_name
    fig.savefig(ysec_path, dpi=run_config.plot_dpi)
    plt.close(fig)
    LOG.info("Saved Y-section plot to %s", ysec_path)


def _resolve_executable(exe_name: str) -> Path | None:
    candidate = Path(exe_name)
    if candidate.is_file():
        return candidate
    resolved = shutil.which(exe_name)
    return Path(resolved) if resolved else None


def _append_grid_lonlat_to_nam(
    nam_path: Path, grid_corners_lonlat: np.ndarray, epsg: int = 4326
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

    assert run_config.grid_output_path.exists(), (
        f"Grid NPZ not found: {run_config.grid_output_path}. "
        f"Run: python build_modflow_grid.py --config {run_config.config_path}"
    )
    assert run_config.material_parameters_path.exists(), (
        f"Material parameters NPZ not found: {run_config.material_parameters_path}. "
        f"Run: python add_material_parameters.py --config {run_config.config_path}"
    )

    grid_data = _grid_arrays_from_npz(run_config.grid_output_path)
    model_data = _grid_arrays_from_npz(run_config.material_parameters_path)

    el_dims = np.asarray(grid_data["el_dims"], dtype=int)
    assert el_dims.size == 3, "el_dims must contain 3 values"
    nx = int(el_dims[0])
    ny = int(el_dims[1])
    nz = int(el_dims[2])
    step = np.asarray(grid_data["step"], dtype=float)
    assert step.size == 3, "step must contain 3 values"

    active_mask = grid_data["active_mask"].astype(bool)
    assert active_mask.shape == (ny, nx), "active_mask shape mismatch"

    top = np.asarray(model_data["top"], dtype=float)
    botm = np.asarray(model_data["botm"], dtype=float)
    materials = np.asarray(model_data["materials"], dtype=int)
    idomain = np.asarray(model_data["idomain"], dtype=int)
    kh = np.asarray(model_data["kh"] if "kh" in model_data else model_data["hk"], dtype=float)
    kv = np.asarray(model_data["kv"] if "kv" in model_data else model_data.get("k33", kh), dtype=float)
    assert top.shape == (ny, nx), "top shape mismatch"
    assert botm.shape == (nz, ny, nx), "botm shape mismatch"
    assert materials.shape == (nz, ny, nx), "materials shape mismatch"
    assert idomain.shape == (nz, ny, nx), "idomain shape mismatch"
    assert kh.shape == (nz, ny, nx), "kh shape mismatch"
    assert kv.shape == (nz, ny, nx), "kv shape mismatch"

    grid_corners_lonlat = grid_data.get("grid_corners_lonlat")
    assert grid_corners_lonlat is not None, "grid_corners_lonlat missing from grid data"
    lonlat_epsg = int(grid_data.get("lonlat_epsg", 4326))

    workdir = run_config.workspace
    workdir.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(
        sim_name=run_config.sim_name,
        exe_name=str(exe_path),
        sim_ws=str(workdir),
    )

    perioddata = [(float(days), 1, 1.0) for days in run_config.stress_periods_days]
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(perioddata),
        perioddata=perioddata,
    )
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=run_config.sim_name, save_flows=True)

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nz,
        nrow=ny,
        ncol=nx,
        delr=float(step[0]),
        delc=float(step[1]),
        top=top,
        botm=botm,
        idomain=idomain,
    )

    # MF6 expects starting heads per layer; use layer tops with optional offset.
    layer_tops = _layer_tops_from_top_botm(top, botm)
    if run_config.initial_head_offset_override is not None:
        offset = float(run_config.initial_head_offset_override)
    elif "initial_head_offset" in model_data:
        offset = float(_scalar_from_model_data(model_data, "initial_head_offset"))
    else:
        offset = 0.0
    layer_tops_for_ic = layer_tops + offset if offset != 0.0 else layer_tops
    flopy.mf6.ModflowGwfic(gwf, strt=layer_tops_for_ic)

    icelltype = np.zeros(nz, dtype=int)
    icelltype[0] = 1
    specific_yield = float(model_data["specific_yield"]) if "specific_yield" in model_data else 0.1
    specific_storage = float(model_data["specific_storage"]) if "specific_storage" in model_data else 1.0e-5
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=icelltype,
        ss=specific_storage,
        sy=specific_yield,
        transient={iper: True for iper in range(len(run_config.stress_periods_days))},
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=kh, k33=kv, save_specific_discharge=True)

    if run_config.recharge_series_m_per_day is not None:
        recharge_rates = run_config.recharge_series_m_per_day
    else:
        if run_config.recharge_rate_override is not None:
            recharge_rate = float(run_config.recharge_rate_override)
        else:
            recharge_rate = float(model_data["recharge_rate"]) if "recharge_rate" in model_data else run_config.recharge_rate
        recharge_rates = tuple([recharge_rate] * len(run_config.stress_periods_days))

    top_active = active_mask & (idomain[0] > 0)
    top_rows, top_cols = np.where(top_active)
    if run_config.use_uzf:
        assert all(rate >= 0.0 for rate in recharge_rates), (
            "UZF finf must be non-negative. Use simulate_et for ET or switch to RCH for signed recharge."
        )
        vks = float(_scalar_from_model_data(model_data, "vks"))
        thtr = float(_scalar_from_model_data(model_data, "thtr"))
        thts = float(_scalar_from_model_data(model_data, "thts"))
        thti = float(_scalar_from_model_data(model_data, "thti"))
        eps = float(_scalar_from_model_data(model_data, "eps"))
        surfdep = float(_scalar_from_model_data(model_data, "surfdep"))
        pet = float(_scalar_from_model_data(model_data, "pet"))
        extdp = float(_scalar_from_model_data(model_data, "extdp"))
        extwc = float(_scalar_from_model_data(model_data, "extwc"))
        ha = float(_scalar_from_model_data(model_data, "ha"))
        hroot = float(_scalar_from_model_data(model_data, "hroot"))
        rootact = float(_scalar_from_model_data(model_data, "rootact"))
        ntrailwaves = int(_scalar_from_model_data(model_data, "ntrailwaves", cast=int))
        nwavesets = int(_scalar_from_model_data(model_data, "nwavesets", cast=int))
        simulate_et = bool(_scalar_from_model_data(model_data, "simulate_et", cast=bool))
        unsat_etwc = bool(_scalar_from_model_data(model_data, "unsat_etwc", cast=bool))
        unsat_etae = bool(_scalar_from_model_data(model_data, "unsat_etae", cast=bool))
        simulate_gwseep = bool(_scalar_from_model_data(model_data, "simulate_gwseep", cast=bool))

        iuzno = np.arange(top_rows.size, dtype=int)
        packagedata = [
            (
                int(iuzno_idx),
                (0, int(row), int(col)),
                1,
                0,
                surfdep,
                vks,
                thtr,
                thts,
                thti,
                eps,
            )
            for iuzno_idx, (row, col) in enumerate(zip(top_rows.tolist(), top_cols.tolist()))
        ]
        perioddata = {}
        for iper, rate in enumerate(recharge_rates):
            finf = float(rate)
            perioddata[iper] = [
                (int(iuzno_idx), finf, pet, extdp, extwc, ha, hroot, rootact)
                for iuzno_idx in iuzno.tolist()
            ]

        flopy.mf6.ModflowGwfuzf(
            gwf,
            nuzfcells=len(packagedata),
            ntrailwaves=ntrailwaves,
            nwavesets=nwavesets,
            packagedata=packagedata,
            perioddata=perioddata,
            simulate_et=simulate_et,
            simulate_gwseep=simulate_gwseep,
            unsat_etwc=unsat_etwc,
            unsat_etae=unsat_etae,
            save_flows=True,
        )
    else:
        recharge_spd = {
            iper: np.where(active_mask, float(rate), 0.0) for iper, rate in enumerate(recharge_rates)
        }
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge_spd)

    if run_config.use_drn:
        drain_cells = []
        if run_config.drain_mode == "top":
            for row, col in zip(top_rows.tolist(), top_cols.tolist()):
                drain_cells.append(
                    (0, int(row), int(col), float(top[row, col]), run_config.drain_conductance)
                )
        elif run_config.drain_mode == "north_side":
            north_row = 0
            for lay in range(nz):
                layer_top = top if lay == 0 else botm[lay - 1]
                cols = np.where(idomain[lay, north_row, :] > 0)[0]
                for col in cols.tolist():
                    drain_cells.append(
                        (
                            int(lay),
                            int(north_row),
                            int(col),
                            float(layer_top[north_row, col]),
                            run_config.drain_conductance,
                        )
                    )
        else:
            raise AssertionError(f"Unsupported drain mode: {run_config.drain_mode}")
        flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drain_cells)

    well_spd = _build_well_spd(
        run_config.wells,
        grid_data,
        top,
        botm,
        idomain,
        len(run_config.stress_periods_days),
    )
    if well_spd:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=well_spd)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{run_config.sim_name}.hds",
        budget_filerecord=f"{run_config.sim_name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    sim.write_simulation()
    _append_grid_lonlat_to_nam(workdir / "mfsim.nam", grid_corners_lonlat, lonlat_epsg)
    _append_grid_lonlat_to_nam(
        workdir / f"{run_config.sim_name}.nam", grid_corners_lonlat, lonlat_epsg
    )
    success, buffer = sim.run_simulation()
    assert success, "Modflow6 did not terminate successfully"
    LOG.info("Modflow6 run complete")
    if buffer:
        LOG.debug("Modflow6 output: %s", "\n".join(buffer))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Modflow6 from prebuilt grid and material parameter files."
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
    build_and_run(args.config, args.workspace)


if __name__ == "__main__":
    main()
