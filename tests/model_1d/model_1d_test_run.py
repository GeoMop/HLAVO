import os
import sys
import numpy as np
import xarray as xr
import pytest
from pathlib import Path

from hlavo.composed.common_data import ComposedData

from hlavo.kalman.model_1d import Model1D
from hlavo.misc.aux_zarr_fuse import load_dotenv
from hlavo.misc.config import load_config
from tests.ingress.scrapper.test_process_one_meteo_raw import workdir

script_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _test_measurements_dataset():
    date_time = np.array(
        [
            "2025-03-06T00:00:00",
            "2025-03-06T02:00:00",
            "2025-03-18T00:00:00",
            "2025-03-18T01:00:00",
            "2025-04-16T03:00:00",
        ],
        dtype="datetime64[ns]",
    )
    site_id = np.array([1], dtype=np.int32)
    depth_level = np.arange(5, dtype=np.int32)
    moisture = np.full((date_time.size, site_id.size, depth_level.size), 0.2)

    return xr.Dataset(
        data_vars={
            "moisture": (("date_time", "site_id", "depth_level"), moisture),
            "longitude": (("date_time", "site_id"), np.full((date_time.size, site_id.size), 14.889853)),
            "latitude": (("date_time", "site_id"), np.full((date_time.size, site_id.size), 50.863565)),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
            "depth_level": depth_level,
        },
    )


def _test_meteo_dataset():
    date_time = np.array(
        ["2025-03-06T00:00:00", "2025-03-06T01:00:00", "2025-03-06T02:00:00"],
        dtype="datetime64[ns]",
    )
    site_id = np.array([1], dtype=np.int32)
    shape = (date_time.size, site_id.size)

    return xr.Dataset(
        data_vars={
            "precipitation": (("date_time", "site_id"), np.zeros(shape)),
            "temperature": (("date_time", "site_id"), np.full(shape, 273.15)),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
        },
    )


def build_model():
    """
    Shared factory usable by pytest fixtures AND __main__.
    """
    load_dotenv()   # load zarr secrets
    cfg, cfg_path = load_config(script_dir / "composed_config.yaml")
    workdir = script_dir / "sandbox"
    composed = ComposedData.from_config(workdir, cfg, cfg_path)

    site_id = 1
    model = Model1D.from_config(
        composed,
        site_id=site_id,
        config=cfg['model_1d']
    )

    return model


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model(monkeypatch):
    return build_model()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skip
def test_measurements_preparation(model):
    from filterpy.kalman import UnscentedKalmanFilter
    from hlavo.kalman.parallel_ukf import ParallelUKF

    assert isinstance(model.ukf, (UnscentedKalmanFilter, ParallelUKF))
    assert isinstance(model.data.profiles_dataset, xr.Dataset)
    assert np.all(model.measurements_dataset["site_id"].values == model.site_id)

@pytest.mark.skip
def test_measurement_for_time(model):
    start_datetime = np.datetime64("2025-03-18T01:00:00")
    end_datetime = np.datetime64("2025-04-16T03:00:00")

    meas = model.get_dataset_values_for_time(model.measurements_dataset, start_datetime, end_datetime)
    times = meas["date_time"].values

    assert times.min() >= start_datetime
    assert times.max() <= end_datetime

@pytest.mark.skip
def test_get_long_lat(model):
    target = np.datetime64("2025-03-18T00:00:00")
    lon, lat = model.get_long_lat(target)

    assert lon == 14.889853
    assert lat == 50.863565


def test_step(model):
    start = np.datetime64("2025-03-06T00:00:00")
    target = np.datetime64("2025-03-06T02:00:00")

    velocity = model.step(start, target, pressure_at_bottom=1.0)
    model.save_results()

    assert not np.isnan(velocity)


# ---------------------------------------------------------------------------
# __main__ support (manual execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running tests manually...\n")

    model = build_model()

    test_measurements_preparation(model)
    print("✓ measurements_preparation passed")

    test_measurement_for_time(model)
    print("✓ measurement_for_time passed")

    test_get_long_lat(model)
    print("✓ get_long_lat passed")

    test_step(model)
    print("✓ step passed")


    print("\nAll tests passed.")
