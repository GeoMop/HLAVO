from dask.distributed import get_client, Queue
from datetime import datetime, timedelta
from hlavo.kalman.kalman import KalmanFilter
from hlavo.ingress.moist_profile.load_data import load_pr2_data, load_odyssey_data, preprocess_data, get_measurements, get_precipitations, load_data
from hlavo.ingress.moist_profile.load_zarr_data import load_zarr_data
from bisect import bisect_left, bisect_right

# ---------------------------------------------------------------------------
# 3D model class
# ---------------------------------------------------------------------------

class Model3D:
    def __init__(self, n_1d, initial_state=0.0, initial_time=0.0, base_dt=timedelta(minutes=5)):
        self.n_1d = n_1d
        self.state = initial_state
        #self.time = initial_time
        self.base_dt = base_dt

    # def choose_dt(self, current_time, t_end):
    #     remaining = t_end - current_time
    #     return max(min(self.base_dt, remaining), 0.0)

    def choose_dt(self, current_time, t_end):
        remaining = t_end - current_time

        if remaining <= timedelta(0):
            return timedelta(0)

        return min(self.base_dt, remaining)

    def step(self, target_time, contributions):
        #@TODO: how to handle this method?
        print(f"[3D] step to t={target_time}, "
              f"current_state={self.state}, contributions={contributions}")
        print("contributions ", contributions)

        #total_contrib = sum(contributions)
        #print("total contribution ", total_contrib)
        #self.state += total_contrib
        #self.time = target_time
        #print(f"[3D] new state={self.state}")
        return self.state

    def run_loop(self, start_datetime, end_datetime, queue_names_out_to_1d, queue_name_in_from_1d):
        client = get_client()
        q_3d_to_1d = [Queue(name) for name in queue_names_out_to_1d]
        q_1d_to_3d = Queue(queue_name_in_from_1d)

        current_time = start_datetime

        print("current time ", current_time)
        print("end date time ", end_datetime)

        while current_time < end_datetime:
            dt = self.choose_dt(current_time, end_datetime)
            print("dt ", dt)
            if dt <= timedelta(minutes=0):
                print("[3D] dt <= 0, stopping to avoid infinite loop.")
                break

            target_time = current_time + dt
            print(f"\n[3D] === Step: t={current_time} -> t={target_time} ===")
            print(f"[3D] current state={self.state}")

            # send to 1D
            for i in range(self.n_1d):
                data_for_i = self.state + i  # dummy placeholder
                print(f"[3D] sending to 1D {i}: t={target_time}, data={data_for_i}")
                q_3d_to_1d[i].put((target_time, data_for_i))

            # receive contributions
            contributions = [None] * self.n_1d
            received = 0

            while received < self.n_1d:
                idx, t_recv, contrib = q_1d_to_3d.get()
                print(f"[3D] received from 1D {idx}: t={t_recv}, contrib={contrib}")

                if contrib is not None:
                    contrib_velocity, contrib_long, contrib_lat = contrib
                    contributions[idx] = contrib_velocity
                received += 1

            self.step(target_time, contributions)
            current_time = target_time

        print(f"[3D] finished time loop at t={current_time} (t_end={end_datetime}), state={self.state}")
        return self.state
