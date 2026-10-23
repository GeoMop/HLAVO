"""Fixtures of the composed-model tests."""
import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture
def dask_client():
    """Client of an in-process Dask cluster (2 single-threaded workers) running the 1D workers.

    Threads instead of processes keep the test fast and the logs in one stream. Cluster and
    client are closed by their context managers when the test ends, also when it fails.
    """
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=False) as cluster, Client(cluster) as client:
        yield client
