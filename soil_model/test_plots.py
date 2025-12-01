import numpy as np
import pytest
import matplotlib.pyplot as plt
from plots import covariance_plot

def generate_random_cov_matrix(dim, seed=42):
    """
    Generates a random symmetric positive semi-definite covariance matrix.
    """
    np.random.seed(seed)
    A = np.random.randn(dim, dim)
    cov_matrix = np.dot(A, A.T)  # Ensures positive semi-definiteness
    return cov_matrix



class StateComponent:
    def __init__(self, size):
        self._size = size
    def size(self):
        return self._size



def test_plot_covariance():
    """
    Test if plot_covariance_analysis runs without errors.
    """
    # Generate a 5x5 random covariance matrix
    state_struct = {
        "Position": StateComponent(10),
        "vG_Ks": StateComponent(2),
        "vG_n": StateComponent(1)
    }

    # Generate random covariance matrix of appropriate size
    dim = sum(obj.size() for obj in state_struct.values())
    random_cov_matrix = np.random.randn(dim, dim)
    random_cov_matrix = random_cov_matrix @ random_cov_matrix.T  # Make it positive semi-definite

    # Call the function
    covariance_plot(random_cov_matrix, time=10, state_struct=state_struct, n_evec=5, show=True)
    plt.close('all')  # Close plots after testing
