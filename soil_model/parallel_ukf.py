import os
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from multiprocessing import Pool, cpu_count


def _fx_wrapper(args):
    print(f"Processing in PID: {os.getpid()}")
    fx, sigma, dt, fx_args = args
    return fx(sigma, dt, **fx_args)


class ParallelUKF(UnscentedKalmanFilter):
    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        Parallel version of compute_process_sigmas.
        Computes the propagated sigma points (sigmas_f) in parallel.
        """
        if fx is None:
            fx = self.fx

        # Calculate sigma points for the current state and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        # Prepare arguments for parallel processing
        args = [(fx, sigmas[i], dt, fx_args) for i in range(sigmas.shape[0])]

        # Run fx in parallel on sigma points
        with Pool(processes=cpu_count()) as pool:
            self.sigmas_f = np.array(pool.map(_fx_wrapper, args))
