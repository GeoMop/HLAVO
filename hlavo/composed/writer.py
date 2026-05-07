from __future__ import annotations

import json
import logging
from pathlib import Path

import attrs
import numpy as np
import polars as pl
import zarr_fuse

from hlavo.composed.common_data import ComposedData

LOG = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[2]
DEFAULT_SIMULATION_SCHEMA_PATH = ROOT_PATH / "hlavo/schemas/simulation_schema.yaml"
DEFAULT_WELLS_SCHEMA_PATH = ROOT_PATH / "hlavo/ingress/well_data/wells_schema.yaml"


@attrs.define(frozen=True)
class WellMetadata:
    well_id: str
    longitude: float
    latitude: float
    z: float | None
    interval_min: float | None
    interval_max: float | None


def _store_kwargs(store_url: str | Path | None) -> dict[str, str]:
    if store_url is None:
        return {}
    return {"STORE_URL": str(store_url)}


def _as_minutes(date_time: np.datetime64) -> np.datetime64:
    return np.datetime64(date_time, "m")


def _calibration_label(date_time: np.datetime64) -> str:
    return np.datetime_as_string(_as_minutes(date_time), unit="m", timezone="UTC")


def load_well_metadata(
    *,
    schema_path: Path = DEFAULT_WELLS_SCHEMA_PATH,
    store_url: str | Path | None = None,
) -> tuple[WellMetadata, ...]:
    root = zarr_fuse.open_store(schema_path, **_store_kwargs(store_url))
    dataset = root["Uhelna"]["water_levels"].dataset
    well_ids = [str(value) for value in dataset["well_id"].values.tolist()]
    metadata: list[WellMetadata] = []
    for well_id in well_ids:
        item = dataset.sel(well_id=well_id)
        longitude = float(np.asarray(item["longitude"]).reshape(-1)[0])
        latitude = float(np.asarray(item["latitude"]).reshape(-1)[0])
        z_value = np.asarray(item["Z"]).reshape(-1)[0] if "Z" in item else np.nan
        interval_min = np.nan
        interval_max = np.nan
        if "interval_min" in item:
            interval_min_values = np.asarray(item["interval_min"], dtype=float).reshape(-1)
            interval_min = float(interval_min_values[0]) if interval_min_values.size > 0 else np.nan
        if "interval_max" in item:
            interval_max_values = np.asarray(item["interval_max"], dtype=float).reshape(-1)
            interval_max = float(interval_max_values[0]) if interval_max_values.size > 0 else np.nan
        metadata.append(
            WellMetadata(
                well_id=well_id,
                longitude=longitude,
                latitude=latitude,
                z=None if np.isnan(z_value) else float(z_value),
                interval_min=None if np.isnan(interval_min) else float(interval_min),
                interval_max=None if np.isnan(interval_max) else float(interval_max),
            )
        )
    LOG.info("Loaded %s monitoring wells for composed output.", len(metadata))
    return tuple(metadata)


@attrs.define
class NullModel3DWriter:
    composed: ComposedData
    calibration: str
    well_metadata: tuple[WellMetadata, ...] = ()

    @classmethod
    def from_config(
        cls,
        *,
        composed: ComposedData,
        locations_1d: list[int],
        writer_config: dict,
        calibration_time: np.datetime64,
    ) -> "NullModel3DWriter":
        _ = locations_1d
        _ = writer_config
        return cls(
            composed=composed,
            calibration=_calibration_label(calibration_time),
        )

    def write_site_step(self, date_time: np.datetime64, site_rows: list[dict]) -> None:
        _ = date_time
        _ = site_rows

    def write_well_step(self, date_time: np.datetime64, well_levels: dict[str, float]) -> None:
        _ = date_time
        _ = well_levels

    def close(self) -> None:
        return None


@attrs.define
class JsonlModel3DWriter(NullModel3DWriter):
    output_path: Path | None = None
    _handle: object | None = None

    @classmethod
    def from_config(
        cls,
        *,
        composed: ComposedData,
        locations_1d: list[int],
        writer_config: dict,
        calibration_time: np.datetime64,
    ) -> "JsonlModel3DWriter":
        _ = locations_1d
        output_path = composed.workdir / str(writer_config.get("jsonl_path", "composed_output.jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8")
        metadata = load_well_metadata(
            schema_path=Path(str(writer_config.get("wells_schema_path", DEFAULT_WELLS_SCHEMA_PATH))),
            store_url=writer_config.get("wells_store_url"),
        )
        writer = cls(
            composed=composed,
            calibration=_calibration_label(calibration_time),
            well_metadata=metadata,
            output_path=output_path,
        )
        writer._handle = handle
        return writer

    def _write_rows(self, *, kind: str, rows: list[dict]) -> None:
        assert self._handle is not None, "JSONL writer handle is not open"
        for row in rows:
            payload = {"kind": kind, **row}
            self._handle.write(json.dumps(payload, default=str) + "\n")

    def write_site_step(self, date_time: np.datetime64, site_rows: list[dict]) -> None:
        _ = date_time
        self._write_rows(kind="site_prediction", rows=site_rows)

    def write_well_step(self, date_time: np.datetime64, well_levels: dict[str, float]) -> None:
        _ = date_time
        rows = []
        for meta in self.well_metadata:
            level = float(well_levels[meta.well_id])
            rows.append(
                {
                    "date_time": str(_as_minutes(date_time)),
                    "well_id": meta.well_id,
                    "calibration": self.calibration,
                    "water_level": level,
                    "water_depth": None if meta.z is None else float(meta.z - level),
                    "longitude": meta.longitude,
                    "latitude": meta.latitude,
                    "Z": meta.z,
                }
            )
        self._write_rows(kind="well_prediction", rows=rows)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@attrs.define
class ZarrModel3DWriter(NullModel3DWriter):
    site_node: object | None = None
    well_node: object | None = None

    @classmethod
    def from_config(
        cls,
        *,
        composed: ComposedData,
        locations_1d: list[int],
        writer_config: dict,
        calibration_time: np.datetime64,
    ) -> "ZarrModel3DWriter":
        _ = locations_1d
        simulation_schema_path = Path(str(writer_config.get("simulation_schema_path", DEFAULT_SIMULATION_SCHEMA_PATH)))
        root = zarr_fuse.open_store(
            simulation_schema_path,
            **_store_kwargs(writer_config.get("simulation_store_url")),
        )
        well_metadata = load_well_metadata(
            schema_path=Path(str(writer_config.get("wells_schema_path", DEFAULT_WELLS_SCHEMA_PATH))),
            store_url=writer_config.get("wells_store_url"),
        )
        return cls(
            composed=composed,
            calibration=_calibration_label(calibration_time),
            well_metadata=well_metadata,
            site_node=root["Uhelna"]["site_prediction"],
            well_node=root["Uhelna"]["well_prediction"],
        )

    def write_site_step(self, date_time: np.datetime64, site_rows: list[dict]) -> None:
        assert self.site_node is not None, "Site prediction node is not open"
        rows = []
        for row in site_rows:
            rows.append(
                {
                    "date_time": str(_as_minutes(date_time)),
                    "site_id": int(row["site_id"]),
                    "calibration": self.calibration,
                    "velocity": float(row["velocity"]),
                    "preasure_head": float(row["pressure_head"]),
                    "longitude": float(row["longitude"]),
                    "latitude": float(row["latitude"]),
                }
            )
        self.site_node.update(pl.DataFrame(rows))

    def write_well_step(self, date_time: np.datetime64, well_levels: dict[str, float]) -> None:
        assert self.well_node is not None, "Well prediction node is not open"
        rows = []
        for meta in self.well_metadata:
            level = float(well_levels[meta.well_id])
            rows.append(
                {
                    "date_time": str(_as_minutes(date_time)),
                    "well_id": meta.well_id,
                    "calibration": self.calibration,
                    "water_level": level,
                    "water_depth": np.nan if meta.z is None else float(meta.z - level),
                    "longitude": float(meta.longitude),
                    "latitude": float(meta.latitude),
                    "Z": np.nan if meta.z is None else float(meta.z),
                }
            )
        self.well_node.update(pl.DataFrame(rows))
