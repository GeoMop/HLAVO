from hlavo.composed.model_1d import Model1D
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dask.distributed import Client, LocalCluster, get_client, Queue


if __name__ == "__main__":
    work_dir = Path(os.getcwd())
    config_file = work_dir / '../../runs/deep_model/deep_model_config.yaml'
    deep_model_config_path = Path(config_file).resolve()

    deep_model_config = load_config(deep_model_config_path)

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    start_datetime = datetime.fromisoformat(deep_model_config["start_datetime"])
    end_datetime = datetime.fromisoformat(deep_model_config["end_datetime"])



    Model1D()