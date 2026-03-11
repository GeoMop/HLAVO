from __future__ import annotations

from typing import Any

import numpy as np
from attrs import frozen


@frozen(slots=True)
class Data3DTo1D:
    date_time: np.datetime64
    dt: np.timedelta64
    site_id: int
    pressure_head: float

    @staticmethod
    def build(date_time: np.datetime64, dt: np.timedelta64, site_id: int, pressure_head: Any) -> "Data3DTo1D":
        return Data3DTo1D(
            date_time=as_datetime64_ms(date_time, "date_time"),
            dt=as_timedelta64_ms(dt, "dt"),
            site_id=int(site_id),
            pressure_head=float(pressure_head),
        )


@frozen(slots=True)
class Data1DTo3D:
    date_time: np.datetime64
    site_id: int
    velocity: float

    @staticmethod
    def build(date_time: np.datetime64, site_id: int, velocity: Any) -> "Data1DTo3D":
        return Data1DTo3D(
            date_time=as_datetime64_ms(date_time, "date_time"),
            site_id=int(site_id),
            velocity=float(velocity),
        )


def as_datetime64_ms(value: Any, value_name: str) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ms]")
    raise TypeError(f"{value_name} must be np.datetime64, got {type(value)}")


def as_timedelta64_ms(value: Any, value_name: str) -> np.timedelta64:
    if isinstance(value, np.timedelta64):
        return value.astype("timedelta64[ms]")
    raise TypeError(f"{value_name} must be np.timedelta64, got {type(value)}")

