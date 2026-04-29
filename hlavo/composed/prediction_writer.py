from __future__ import annotations

import json
import os
from pathlib import Path

import attrs
import numpy as np
import xarray as xr
import zarr_fuse as zf

from hlavo.composed.common_data import ComposedData
from hlavo.composed.data_1d_to_3d import Data1DTo3D
from hlavo.misc.class_resolve import resolve_named_class


def _calibration_label(date_time: np.datetime64) -> str:
    return str(np.datetime64(date_time, "m"))


def _schema_path(composed: ComposedData, path: str | Path) -> Path:
    return composed.relative_resolve(path)


@attrs.define
class PredictionWriterBase:
    composed: ComposedData
    locations_1d: list[int]
    calibration: str
    wells_dataset: xr.Dataset | None

    @property
    def wells(self) -> xr.Dataset | None:
        return self.wells_dataset

    def write_step(
        self,
        date_time: np.datetime64,
        site_messages: list[Data1DTo3D],
        pressure_heads: dict[int, float],
        well_prediction: dict[str, float],
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


@attrs.define
class FilePredictionWriter(PredictionWriterBase):
    output_path: Path

    @classmethod
    def from_config(cls, composed: ComposedData, locations_1d: list[int], config: dict):
        output_path = composed.workdir / config["file_name"]
        wells_dataset = _load_wells_dataset(composed, config)
        return cls(
            composed=composed,
            locations_1d=locations_1d,
            calibration=_calibration_label(composed.start),
            wells_dataset=wells_dataset,
            output_path=output_path,
        )

    def write_step(
        self,
        date_time: np.datetime64,
        site_messages: list[Data1DTo3D],
        pressure_heads: dict[int, float],
        well_prediction: dict[str, float],
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            for msg in site_messages:
                row = {
                    "node": "site_prediction",
                    "date_time": str(np.datetime64(date_time, "s")),
                    "site_id": int(msg.site_id),
                    "calibration": self.calibration,
                    "velocity": float(msg.velocity),
                    "pressure_head": float(pressure_heads[int(msg.site_id)]),
                    "longitude": float(msg.longitude),
                    "latitude": float(msg.latitude),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            for well_id, water_level in well_prediction.items():
                row = {
                    "node": "well_prediction",
                    "date_time": str(np.datetime64(date_time, "s")),
                    "well_id": str(well_id),
                    "calibration": self.calibration,
                    "water_level": float(water_level),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")


@attrs.define
class ZarrPredictionWriter(PredictionWriterBase):
    site_node: object
    well_node: object
    site_datasets: list[xr.Dataset] = attrs.field(factory=list)
    well_datasets: list[xr.Dataset] = attrs.field(factory=list)

    @classmethod
    def from_config(cls, composed: ComposedData, locations_1d: list[int], config: dict):
        schema_path = _schema_path(composed, config["schema_file"])
        store_url = None
        if "store_url" in config:
            store_url = str(composed.relative_resolve(config["store_url"]))
        root = _open_store(schema_path, store_url)
        wells_dataset = _load_wells_dataset(composed, config)
        return cls(
            composed=composed,
            locations_1d=locations_1d,
            calibration=_calibration_label(composed.start),
            wells_dataset=wells_dataset,
            site_node=root["Uhelna"]["site_prediction"],
            well_node=root["Uhelna"]["well_prediction"],
        )

    def write_step(
        self,
        date_time: np.datetime64,
        site_messages: list[Data1DTo3D],
        pressure_heads: dict[int, float],
        well_prediction: dict[str, float],
    ) -> None:
        self.site_datasets.append(self._site_dataset(date_time, site_messages, pressure_heads))
        if well_prediction:
            self.well_datasets.append(self._well_dataset(date_time, well_prediction))

    def close(self) -> None:
        if self.site_datasets:
            self.site_node.write_ds(_concat_time(self.site_datasets), mode="w")
        if self.well_datasets:
            self.well_node.write_ds(_concat_time(self.well_datasets), mode="w")

    def _site_dataset(
        self,
        date_time: np.datetime64,
        site_messages: list[Data1DTo3D],
        pressure_heads: dict[int, float],
    ) -> xr.Dataset:
        site_id = np.array([int(msg.site_id) for msg in site_messages], dtype=np.int32)
        date_times = np.array([date_time], dtype="datetime64[m]")
        calibration = np.array([self.calibration], dtype="U16")
        shape = (date_times.size, site_id.size, calibration.size)
        velocity = np.array([[float(msg.velocity) for msg in site_messages]], dtype=float)[:, :, None]
        pressure_head = np.array(
            [[float(pressure_heads[int(msg.site_id)]) for msg in site_messages]],
            dtype=float,
        )[:, :, None]
        assert velocity.shape == shape
        assert pressure_head.shape == shape
        return xr.Dataset(
            data_vars={
                "velocity": (("date_time", "site_id", "calibration"), velocity),
                "pressure_head": (("date_time", "site_id", "calibration"), pressure_head),
                "longitude": (("site_id",), np.array([float(msg.longitude) for msg in site_messages])),
                "latitude": (("site_id",), np.array([float(msg.latitude) for msg in site_messages])),
                "Z": (("site_id",), np.full(site_id.size, np.nan)),
            },
            coords={
                "date_time": date_times,
                "site_id": site_id,
                "calibration": calibration,
            },
        )

    def _well_dataset(self, date_time: np.datetime64, well_prediction: dict[str, float]) -> xr.Dataset:
        well_id = np.array(list(well_prediction), dtype="U16")
        date_times = np.array([date_time], dtype="datetime64[m]")
        calibration = np.array([self.calibration], dtype="U16")
        shape = (date_times.size, well_id.size, calibration.size)
        water_level = np.array([[float(value) for value in well_prediction.values()]], dtype=float)[:, :, None]
        assert water_level.shape == shape
        water_depth = np.full(shape, np.nan)
        longitude, latitude, z = _well_coordinates(self.wells_dataset, well_id)
        return xr.Dataset(
            data_vars={
                "water_level": (("date_time", "well_id", "calibration"), water_level),
                "water_depth": (("date_time", "well_id", "calibration"), water_depth),
                "longitude": (("well_id",), longitude),
                "latitude": (("well_id",), latitude),
                "Z": (("well_id",), z),
            },
            coords={
                "date_time": date_times,
                "well_id": well_id,
                "calibration": calibration,
            },
        )


def prediction_writer_from_config(composed: ComposedData, locations_1d: list[int], config: dict):
    writer_config = config.get("writer")
    if writer_config is None:
        return None
    assert isinstance(writer_config, dict), "model_3d.common.writer must be a mapping"
    writer_class = resolve_named_class(
        writer_config["class_name"],
        (FilePredictionWriter, ZarrPredictionWriter),
    )
    return writer_class.from_config(composed, locations_1d, writer_config)


def _load_wells_dataset(composed: ComposedData, config: dict) -> xr.Dataset | None:
    if "wells_schema_file" not in config:
        return None
    store_url = None
    if "wells_store_url" in config:
        store_url = str(composed.relative_resolve(config["wells_store_url"]))
    root = _open_store(_schema_path(composed, config["wells_schema_file"]), store_url)
    return root["Uhelna"]["water_levels"].dataset.compute()


def _open_store(schema_path: Path, store_url: str | None):
    if store_url is None:
        return zf.open_store(schema_path)

    schema = zf.schema.deserialize(schema_path)
    schema.ds.ATTRS["STORE_URL"] = store_url
    previous_store_url = os.environ.pop("ZF_STORE_URL", None)
    try:
        return zf.open_store(schema)
    finally:
        if previous_store_url is not None:
            os.environ["ZF_STORE_URL"] = previous_store_url


def _well_coordinates(wells_dataset: xr.Dataset | None, well_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if wells_dataset is None:
        nan_values = np.full(well_id.size, np.nan)
        return nan_values, nan_values.copy(), nan_values.copy()

    wells = wells_dataset.sel(well_id=well_id)
    longitude = np.asarray(wells["longitude"].values, dtype=float)
    latitude = np.asarray(wells["latitude"].values, dtype=float)
    z = np.asarray(wells["Z"].values, dtype=float)
    return longitude, latitude, z


def _dataset_values(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    values = {name: np.asarray(dataset.coords[name].values) for name in dataset.coords}
    values.update({name: np.asarray(dataset[name].values) for name in dataset.data_vars})
    return values


def _concat_time(datasets: list[xr.Dataset]) -> xr.Dataset:
    return xr.concat(
        datasets,
        dim="date_time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
