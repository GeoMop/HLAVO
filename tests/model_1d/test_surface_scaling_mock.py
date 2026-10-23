"""SurfaceScalingMock: precipitation converted to m/day from its declared unit."""
import numpy as np
import pytest
import xarray as xr

from hlavo.kalman.model_1d import SurfaceScalingMock


def _meteo(apcp: float, attrs: dict) -> xr.Dataset:
    """One day of hourly APCP with the given value and variable attributes."""
    date_time = np.arange(
        np.datetime64("2025-03-01T00:00", "m"),
        np.datetime64("2025-03-02T00:00", "m"),
        np.timedelta64(1, "h"),
    )
    apcp_da = xr.DataArray(np.full(date_time.size, apcp), dims=["date_time"], coords={"date_time": date_time}, attrs=attrs)
    return xr.Dataset({"APCP": apcp_da})


def _measurements(moisture: float) -> xr.Dataset:
    date_time = np.array(["2025-03-01T06:00"], dtype="datetime64[m]")
    return xr.Dataset(
        {"moisture": (("date_time", "depth_level"), np.full((1, 2), moisture))},
        coords={"date_time": date_time},
    )


def test_precipitation_converted_from_declared_unit():
    mock = SurfaceScalingMock(precipitation_var="APCP")
    # 10 mm/day stored in mm/s, as zarr_fuse writes it from chmi_stations_schema.yaml.
    meteo = _meteo(10.0 / 86400.0, {"unit": "millimeter / second"})
    recharge = mock.kalman_step(None, _measurements(0.2), meteo, -60.0)
    # 10 mm/day = 0.01 m/day; scaling factor 0.1 + 0.7 * mean moisture.
    assert recharge == pytest.approx((0.1 + 0.7 * 0.2) * 0.01)


def test_precipitation_without_unit_fails():
    mock = SurfaceScalingMock(precipitation_var="APCP")
    with pytest.raises(AssertionError, match="unit"):
        mock.kalman_step(None, _measurements(0.2), _meteo(1.0, {}), -60.0)
