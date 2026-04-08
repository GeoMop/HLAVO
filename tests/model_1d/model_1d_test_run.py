import os
import numpy as np
import xarray as xr
import pytest
from datetime import datetime
from pathlib import Path

#from dask.distributed import Client, LocalCluster

from hlavo.composed.model_1d import Model1D
from hlavo.composed.composed_model_mock import relative_to_absolute_paths
from filterpy.kalman import UnscentedKalmanFilter
from hlavo.kalman.parallel_ukf import ParallelUKF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path):
    import yaml
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def build_model():
    """
    Shared factory usable by pytest fixtures AND __main__.
    """

    work_dir = Path(os.getcwd())
    config_file = work_dir / '../../runs/composed_1d_only/composed_1d_part_config_test.yaml'
    composed_model_config_path = Path(config_file).resolve()
    config_dir = composed_model_config_path.parent

    composed_model_config = load_config(composed_model_config_path)

    composed_model_config = relative_to_absolute_paths(composed_model_config, config_dir)

    start_datetime = datetime.fromisoformat(composed_model_config["start_datetime"])
    end_datetime = datetime.fromisoformat(composed_model_config["end_datetime"])

    site_id = 1
    seed = composed_model_config["seed"]

    kalman_config_path = composed_model_config["1d_models"][0]
    model_1d_config = load_config(Path(kalman_config_path).resolve())
    model_1d_config = relative_to_absolute_paths(model_1d_config, config_dir)

    model = Model1D(
        site_id=site_id,
        initial_state=0.0,
        work_dir=work_dir,
        model_kalman_config_dict=model_1d_config,
        seed=seed
    )

    return model


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model(dask_client):
    return build_model()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_measurements_preparation(model):
    ukf = model._prepare_kalman_measurements()

    assert isinstance(ukf, (UnscentedKalmanFilter, ParallelUKF))
    assert isinstance(model.measurements_dataset, xr.Dataset)
    assert np.all(model.measurements_dataset["site_id"].values == model.site_id)


def test_measurement_for_time(model):
    start_datetime = np.datetime64("2025-03-18T01:00:00")
    end_datetime = np.datetime64("2025-04-16T03:00:00")

    meas = model.get_measurement_for_time(start_datetime, end_datetime)
    times = meas["date_time"].values

    assert times.min() >= start_datetime
    assert times.max() <= end_datetime


def test_get_long_lat(model):
    target = np.datetime64("2025-03-18T00:00:00")
    lon, lat = model.get_long_lat(target)

    assert lon == 14.889853
    assert lat == 50.863565


def test_step(model):
    start = np.datetime64("2025-03-06T00:00:00")
    target = np.datetime64("2025-03-06T02:00:00")

    result = model.step(start, target, pressure_at_bottom=1.0)

    assert result is not None
    assert hasattr(result, "velocity")


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
