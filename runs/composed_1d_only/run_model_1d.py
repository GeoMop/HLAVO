import os
import numpy as np
from pathlib import Path
from hlavo.composed.model_1d import Model1D
from hlavo.composed.data_3d_to_1d import Data3DTo1D
from hlavo.composed.composed_model_mock import relative_to_absolute_paths
from dask.distributed import Client, LocalCluster, get_client, Queue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path):
    import yaml
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def build_model(composed_model_config: dict):
    composed_model_config = relative_to_absolute_paths(composed_model_config, config_dir)

    site_id = 1
    seed = composed_model_config["seed"]

    model_1d_config = composed_model_config["model_1d"]
    #model_1d_config = load_config(Path(kalman_config_path).resolve())
    model_1d_config = relative_to_absolute_paths(model_1d_config, config_dir)

    model = Model1D(
        site_id=site_id,
        initial_state=0.0,
        work_dir=work_dir,
        model_kalman_config_dict=model_1d_config,
        seed=seed
    )

    return model

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    queue_name_1d_to_3d = "q-1d-to-3d"
    Queue(queue_name_1d_to_3d, client=client)  # ensure creation

    q_name_3d_to_1d = "q-3d-to-1d-1"
    q_3d_to_1d = Queue(q_name_3d_to_1d, client=client)

    work_dir = Path(os.getcwd())
    config_file = work_dir / 'composed_config.yaml'
    print("config file ", config_file)
    composed_model_config_path = Path(config_file).resolve()
    config_dir = composed_model_config_path.parent

    composed_model_config = load_config(composed_model_config_path)

    start_datetime =  np.datetime64(composed_model_config["start_datetime"])
    end_datetime =  np.datetime64(composed_model_config["end_datetime"])

    data_to_1d = Data3DTo1D(site_id=1, date_time=end_datetime, pressure_head=0)
    q_3d_to_1d.put((end_datetime, data_to_1d))

    model = build_model(composed_model_config)

    model.run_loop(start_datetime, end_datetime, q_name_3d_to_1d, queue_name_1d_to_3d)

    client.close()
    cluster.close()
