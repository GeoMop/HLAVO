from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import attrs
import numpy as np
import yaml

import flopy

from run_model import _groundwater_surface_from_head

LOG = logging.getLogger(__name__)


@attrs.define(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    path: Path
    payload: dict


@attrs.define(frozen=True)
class ScenarioResult:
    run_id: str
    scenario_name: str
    scenario_file: str
    status: str
    started_utc: str
    finished_utc: str
    duration_s: float
    run_dir: str
    workspace: str
    config_path: str
    git_commit: str
    error: str
    recharge_summary: str
    drain_conductance: float
    sand_hk: float
    clay_hk: float
    final_head_mean: float
    final_head_min: float
    final_head_max: float
    groundwater_change_mean: float
    groundwater_change_min: float
    groundwater_change_max: float


def _load_yaml(path: Path) -> dict:
    assert path.exists(), f"Config not found: {path}"
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict), f"YAML must be a mapping: {path}"
    return raw


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _stable_hash(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _git_commit(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _scenario_specs(scenarios_dir: Path) -> list[ScenarioSpec]:
    assert scenarios_dir.exists(), f"Scenarios directory not found: {scenarios_dir}"
    files = sorted(scenarios_dir.glob("*.yaml"))
    assert files, f"No scenario YAML files found in {scenarios_dir}"
    specs: list[ScenarioSpec] = []
    for path in files:
        raw = _load_yaml(path)
        scenario_raw = raw.get("scenario", {})
        assert isinstance(scenario_raw, dict), f"scenario must be a mapping in {path}"
        name = str(scenario_raw.get("name", path.stem)).strip()
        assert name, f"Scenario name cannot be empty in {path}"
        description = str(scenario_raw.get("description", "")).strip()
        payload = dict(raw)
        payload.pop("scenario", None)
        specs.append(ScenarioSpec(name=name, description=description, path=path, payload=payload))
    return specs


def _resolved_paths(config: dict) -> tuple[Path, Path, Path]:
    model_raw = config.get("model", {})
    assert isinstance(model_raw, dict), "model must be a mapping"
    model_name = str(model_raw["model_name"])
    workspace = Path(str(model_raw.get("workspace", "model"))) / model_name

    grid_path = Path(str(config.get("grid_output_path", "grid_materials.npz")))
    if not grid_path.is_absolute():
        grid_path = workspace / grid_path

    material_path = Path(str(config.get("material_parameters_output_path", "material_parameters.npz")))
    if not material_path.is_absolute():
        material_path = workspace / material_path

    return workspace, grid_path, material_path


def _recharge_summary(config: dict) -> str:
    model_raw = config.get("model", {})
    assert isinstance(model_raw, dict), "model must be a mapping"
    series = model_raw.get("recharge_series_m_per_day")
    if isinstance(series, list) and series:
        vals = [float(v) for v in series]
        return f"series(min={min(vals):.6g},max={max(vals):.6g},n={len(vals)})"

    materials_raw = config.get("materials", {})
    if isinstance(materials_raw, dict):
        all_raw = materials_raw.get("all", {})
        if isinstance(all_raw, dict) and "recharge_rate" in all_raw:
            return f"constant({float(all_raw['recharge_rate']):.6g})"

    return f"constant({float(model_raw.get('recharge_rate', 1.0e-4)):.6g})"


def _set_hk(config: dict, set_name: str) -> float:
    materials_raw = config.get("materials", {})
    if not isinstance(materials_raw, dict):
        return float("nan")
    sets_raw = materials_raw.get("sets", {})
    if not isinstance(sets_raw, dict):
        return float("nan")
    set_raw = sets_raw.get(set_name, {})
    if not isinstance(set_raw, dict):
        return float("nan")
    if "horizontal_conductivity" not in set_raw:
        return float("nan")
    return float(set_raw["horizontal_conductivity"])


def _compute_metrics(workspace: Path, sim_name: str, material_npz: Path) -> dict[str, float]:
    head_path = workspace / f"{sim_name}.hds"
    assert head_path.exists(), f"Head file not found: {head_path}"

    hds = flopy.utils.binaryfile.HeadFile(str(head_path))
    times = hds.get_times()
    assert times, "No time steps in head output"
    head_initial = np.asarray(hds.get_data(totim=times[0]), dtype=float)
    head_final = np.asarray(hds.get_data(totim=times[-1]), dtype=float)

    with np.load(material_npz, allow_pickle=True) as data:
        idomain = np.asarray(data["idomain"], dtype=int)
        top = np.asarray(data["top"], dtype=float)
        botm = np.asarray(data["botm"], dtype=float)

    wt_initial = _groundwater_surface_from_head(head_initial, idomain, top, botm)
    wt_final = _groundwater_surface_from_head(head_final, idomain, top, botm)
    wt_change = wt_final - wt_initial

    return {
        "final_head_mean": float(np.nanmean(head_final)),
        "final_head_min": float(np.nanmin(head_final)),
        "final_head_max": float(np.nanmax(head_final)),
        "groundwater_change_mean": float(np.nanmean(wt_change)),
        "groundwater_change_min": float(np.nanmin(wt_change)),
        "groundwater_change_max": float(np.nanmax(wt_change)),
    }


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _run_step(script_dir: Path, step: str, config_path: Path, workspace_override: Path | None, log_path: Path) -> None:
    cmd = [sys.executable, str(script_dir / step), "--config", str(config_path)]
    if workspace_override is not None and step in {"run_model.py", "create_paraview.py", "create_plots.py"}:
        cmd.extend(["--workspace", str(workspace_override)])

    banner = f">>> {' '.join(cmd)}"
    LOG.info(banner)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n{banner}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None, "Missing subprocess stdout pipe"
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = proc.wait()
    assert return_code == 0, f"Step failed ({step}), see log: {log_path}"


def _append_index(index_path: Path, result: ScenarioResult) -> None:
    row = attrs.asdict(result)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _grid_signature(config: dict) -> str:
    relevant = {
        "qgis_project_path": config.get("qgis_project_path"),
        "boundary_layer_name": config.get("boundary_layer_name"),
        "raster_group_name": config.get("raster_group_name"),
        "meshsteps": config.get("meshsteps"),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_scenarios(
    base_config_path: Path,
    scenarios_dir: Path,
    runs_root: Path | None,
    skip_grid: bool,
    reuse_grid: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    base_cfg = _load_yaml(base_config_path)
    model_raw = base_cfg.get("model", {})
    assert isinstance(model_raw, dict), "Base model section missing"
    model_name = str(model_raw["model_name"])

    specs = _scenario_specs(scenarios_dir)
    if runs_root is None:
        runs_root = script_dir / "runs" / model_name
    runs_root.mkdir(parents=True, exist_ok=True)
    index_path = runs_root / "index.csv"
    git_commit = _git_commit(script_dir.parent)
    grid_cache_dir = runs_root / "_grid_cache"
    if reuse_grid:
        grid_cache_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Running %s scenarios for model=%s", len(specs), model_name)

    for spec in specs:
        merged = _deep_merge(base_cfg, spec.payload)
        merged_model_raw = merged.get("model", {})
        assert isinstance(merged_model_raw, dict), "Merged model must be mapping"
        merged_model_raw["model_name"] = model_name

        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg_hash = _stable_hash(merged)
        run_id = f"{stamp}_{spec.name}_{cfg_hash}"
        run_dir = runs_root / run_id
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        workspace_override = run_dir / "workspace"
        resolved_cfg = _deep_merge(merged, {"model": {"workspace": str(workspace_override)}})
        resolved_cfg_path = run_dir / "config_resolved.yaml"
        _write_yaml(resolved_cfg_path, resolved_cfg)
        shutil.copy2(spec.path, run_dir / "scenario.yaml")

        started = dt.datetime.utcnow()
        t0 = time.perf_counter()
        status = "ok"
        error = ""

        final_metrics = {
            "final_head_mean": float("nan"),
            "final_head_min": float("nan"),
            "final_head_max": float("nan"),
            "groundwater_change_mean": float("nan"),
            "groundwater_change_min": float("nan"),
            "groundwater_change_max": float("nan"),
        }

        try:
            log_path = logs_dir / "workflow.log"
            LOG.info("Scenario start: %s (run_id=%s)", spec.name, run_id)
            ws, grid_path, _material_path = _resolved_paths(resolved_cfg)
            grid_sig = _grid_signature(resolved_cfg)
            grid_cache_path = grid_cache_dir / f"{grid_sig}.npz"

            if skip_grid:
                pass
            elif reuse_grid and grid_cache_path.exists():
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(grid_cache_path, grid_path)
            else:
                _run_step(script_dir, "build_modflow_grid.py", resolved_cfg_path, workspace_override, log_path)
                if reuse_grid:
                    assert grid_path.exists(), f"Grid output missing after build: {grid_path}"
                    shutil.copy2(grid_path, grid_cache_path)

            _run_step(script_dir, "add_material_parameters.py", resolved_cfg_path, workspace_override, log_path)
            _run_step(script_dir, "run_model.py", resolved_cfg_path, workspace_override, log_path)
            _run_step(script_dir, "create_paraview.py", resolved_cfg_path, workspace_override, log_path)
            _run_step(script_dir, "create_plots.py", resolved_cfg_path, workspace_override, log_path)

            ws, _grid_path, mat_path = _resolved_paths(resolved_cfg)
            sim_name = str(resolved_cfg["model"]["sim_name"])
            final_metrics = _compute_metrics(ws, sim_name, mat_path)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            LOG.exception("Scenario %s failed", spec.name)

        finished = dt.datetime.utcnow()
        duration_s = float(time.perf_counter() - t0)

        result = ScenarioResult(
            run_id=run_id,
            scenario_name=spec.name,
            scenario_file=str(spec.path.name),
            status=status,
            started_utc=started.isoformat(timespec="seconds") + "Z",
            finished_utc=finished.isoformat(timespec="seconds") + "Z",
            duration_s=duration_s,
            run_dir=str(run_dir),
            workspace=str(workspace_override / model_name),
            config_path=str(resolved_cfg_path),
            git_commit=git_commit,
            error=error,
            recharge_summary=_recharge_summary(resolved_cfg),
            drain_conductance=float(resolved_cfg["model"].get("drain_conductance", 1.0)),
            sand_hk=_set_hk(resolved_cfg, "sands"),
            clay_hk=_set_hk(resolved_cfg, "clays"),
            final_head_mean=final_metrics["final_head_mean"],
            final_head_min=final_metrics["final_head_min"],
            final_head_max=final_metrics["final_head_max"],
            groundwater_change_mean=final_metrics["groundwater_change_mean"],
            groundwater_change_min=final_metrics["groundwater_change_min"],
            groundwater_change_max=final_metrics["groundwater_change_max"],
        )
        _append_index(index_path, result)

        if status == "ok":
            LOG.info("Scenario finished: %s (run_id=%s)", spec.name, run_id)
        else:
            LOG.warning("Scenario %s failed: run_id=%s error=%s", spec.name, run_id, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario batch for one base model config.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/base/with_mine_fine.yaml"),
        help="Path to base model config YAML",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("config/scenarios/with_mine_fine"),
        help="Directory with scenario YAML overlays",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Output root for run folders (default: deep_model/runs/<model_name>)",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip build_modflow_grid step (use existing grid files per run workspace)",
    )
    parser.add_argument(
        "--no-reuse-grid",
        action="store_true",
        help="Disable grid cache/reuse and build grid for every scenario run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_scenarios(
        args.base_config,
        args.scenarios_dir,
        args.runs_root,
        args.skip_grid,
        not args.no_reuse_grid,
    )


if __name__ == "__main__":
    main()
