import numpy as np

def dirichlet_bc(h_target):
    """
    Factory for Dirichlet boundary condition applied in a weak sense.
    Returns a function that calculates the inward flux.

    Parameters:
    h_target : float
        The target pressure head for the boundary.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return -K * (H - h_target) / dz

    return bc_function

def neumann_bc(q):
    """
    Factory for Neumann boundary condition (given flux).
    Returns a function that calculates the inward flux.

    Parameters:
    q : float
        Prescribed water flux.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return q

    return bc_function

def free_drainage_bc():
    """
    Factory for free drainage boundary condition.
    Returns a function that calculates the inward flux.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return K

    return bc_function


def poly_ramp(x, eps=0.01):
    """
    A piecewise cubic C2 function:
    f(x) = 0.0      for x < -eps
    f(x) = C2       -eps, 0
    f(x) = 1.0      for x > 0.0
    """
    xx = 2 * x / eps + 1
    result = np.where(
        x <= -eps,  # Values below -eps
        0.0,
        np.where(
            x >= 0,  # Values above 0
            1.0,
            0.25 * (-xx ** 3 + 3 * xx) + 0.5  # Transition region
        )
    )
    return result

def seepage_bc(q_given, h_crit, transition_width):
    """
    Factory for seepage boundary condition.
    Smooth interpolation between given flux and pressure.

    Parameters:
    q_given : float
        Prescribed flux.
    h_crit : float
        Critical pressure head.
    transition_width : float
        Width of the pressure head transition region for interpolation.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        factor = poly_ramp(H - h_crit)
        dirichlet_term = -100 * K * (H - h_crit) / abs(dz)
        prescribed_flux = q_given
        flux = factor * dirichlet_term + (1 - factor) * prescribed_flux
        return flux
    return bc_function
