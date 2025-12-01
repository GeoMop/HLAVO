# run_map.py
from dask.distributed import Client, performance_report
import time
import os

SCHED_FILE = os.path.join(os.getcwd(), "scheduler.json")

from functools import wraps
from multiprocessing import get_context

def run_in_subprocess(func):
    """
    Decorator: execute the wrapped function in a fresh spawned subprocess.

    Usage:
        @run_in_subprocess
        def my_cpp_func(x, y):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.apply(func, args, kwargs)
    return wrapper

@run_in_subprocess
def mock_calc(x):
    # ~2s CPU-idle task
    time.sleep(2)
    return x * x


def main():
    # Connect to the scheduler that the PBS script starts
    client = Client(scheduler_file=SCHED_FILE)

    n_tasks = 40  # change as you like
    xs = list(range(n_tasks))

    t0 = time.time()
    futures = client.map(mock_calc, xs)
    results = client.gather(futures)
    elapsed = time.time() - t0

    print(f"Computed {len(results)} results in {elapsed:.2f}s")
    print("First 10 results:", results[:10])

    # Optional: write a small HTML performance report
    with performance_report(filename="dask_report.html"):
        client.gather(client.map(mock_calc, xs[:10]))

if __name__ == "__main__":
    main()
