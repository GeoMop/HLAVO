import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from concurrent.futures import ThreadPoolExecutor


def _fx_wrapper(args):
    fx, sigma, dt, fx_args = args
    return fx(sigma, dt, **fx_args)


class ParallelUKF(UnscentedKalmanFilter):

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        Parallel version of compute_process_sigmas using threads instead of processes.
        Compatible with Dask workers (which are daemonic).
        """
        if fx is None:
            fx = self.fx

        # Generate sigma points
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        # Prepare arguments for parallel calls
        args = [(fx, sigmas[i], dt, fx_args) for i in range(sigmas.shape[0])]

        # Use threads, not processes
        with ThreadPoolExecutor(max_workers=None) as pool:
            results = list(pool.map(_fx_wrapper, args))

        self.sigmas_f = np.array(results)
