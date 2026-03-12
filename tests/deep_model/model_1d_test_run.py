from hlavo.composed.model_1d import Model1D
import os
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path
from dask.distributed import Client, LocalCluster, get_client, Queue
from filterpy.kalman import UnscentedKalmanFilter
from hlavo.kalman.parallel_ukf import ParallelUKF


def load_config(config_path):
    import yaml
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def measurements_preparation_test(model):
    ukf = model._prepare_kalman_measurements()

    assert isinstance(ukf, (UnscentedKalmanFilter, ParallelUKF)), "Expected UnscentedKalmanFilter or ParallelUKF"
    assert isinstance(model.measurements_dataset, xr.Dataset), "Expected an xarray.Dataset"
    assert np.all(model.measurements_dataset["site_id"].values == model.site_id)


def measurement_for_time_test(model):
    start_datetime = np.datetime64("2025-03-18T01:00:00")
    end_datetime = np.datetime64("2025-04-16T03:00:00")

    meas = model.get_measurement_for_time(start_datetime, end_datetime)
    times = meas["date_time"].values

    assert times.min() >= start_datetime
    assert times.max() <= end_datetime


def get_long_lat_test(model):
    target = np.datetime64("2025-03-18T00:00:00")
    lon, lat = model.get_long_lat(target)

    assert lon == 14.889853
    assert lat == 50.863565


def step_test(model):
    start = np.datetime64("2025-03-18T00:00:00")
    target = np.datetime64("2025-03-21T00:00:00")

    data_to_3d = model.step(start, target, pressure_at_bottom=np.zeros(10))



if __name__ == "__main__":
    work_dir = Path(os.getcwd())
    config_file = work_dir / '../../runs/deep_model/deep_model_config_test.yaml'
    deep_model_config_path = Path(config_file).resolve()

    deep_model_config = load_config(deep_model_config_path)

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    start_datetime = datetime.fromisoformat(deep_model_config["start_datetime"])
    end_datetime = datetime.fromisoformat(deep_model_config["end_datetime"])

    site_id = 1
    seed = deep_model_config["seed"]
    kalman_config_path = deep_model_config["1d_models"][0]
    model_1d_config_path = Path(kalman_config_path).resolve()
    model = Model1D(site_id=site_id, initial_state=0.0, work_dir=work_dir, kalman_config_path=model_1d_config_path, seed=seed)


    measurements_preparation_test(model)
    measurement_for_time_test(model)
    get_long_lat_test(model)
    step_test(model)
