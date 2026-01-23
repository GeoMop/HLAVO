from typing import *
import numpy as np
import attrs
from functools import cached_property
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

@attrs.define
class VanGenuchtenParams:
    """
    Van Genuchten parameters for a single soil material,
    with an optional 'storativity' that only matters if you want
    to find a 'critical' head h_crit where capacity = storativity.

    Attributes
    ----------
    theta_r      : float
        Residual volumetric water content
    theta_s      : float
        Saturated volumetric water content
    alpha        : float
        van Genuchten alpha parameter [1/L]
    n            : float
        van Genuchten n parameter [-]
    K_s          : float
        Saturated hydraulic conductivity [L/T]
    l            : float
        Pore connectivity parameter (often 0.5)
    storativity  : float
        Compressibility effect; for solving capacity(h_crit)=storativity
    """

    theta_r: float
    theta_s: float
    alpha: float
    n: float
    K_s: float
    l: float = 0.5
    storativity: float = 1e-6   # compressibility of water

    def update(self, **kwargs):
        """
        Return a new VanGenuchtenParams instance that is a copy of self,
        but with any attributes given in kwargs replaced.

        Example usage:

            new_params = old_params.update(theta_r=0.05, alpha=0.2)
        """
        # Grab a dict of all current fields/values
        fields = {field.name: getattr(self, field.name) for field in attrs.fields(self.__class__)}

        # Overwrite any fields that appear in kwargs
        fields.update(kwargs)

        # Construct and return a new instance of the same class
        return self.__class__(**fields)

    # ------------------------------------------------------------
    # Argmax of capacity in terms of x = -h and corresponding h_max_cap
    # ------------------------------------------------------------
    @cached_property
    def h_max_cap(self) -> float:
        """
        h_max_cap = - x_star, where x_star is the location in x=-h where
                    the van Genuchten capacity (without storativity) is maximum.

        x_star = 1/alpha * ((n-1)/n)^(1/n)
        """
        x_star = (1.0 / self.alpha) * ((self.n - 1.0) / self.n) ** (1.0 / self.n)
        return -x_star

    @property
    def m(self):
        return 1.0 - 1.0 / self.n

    # ------------------------------------------------------------
    # 3) "Pure" VG capacity for h<0, ignoring storativity
    # ------------------------------------------------------------
    def capacity_vg_only(self, h: float) -> float:
        """
        The derivative of
          theta_r + (theta_s - theta_r)*Se(h)
        w.r.t. h, for h<0. If h>=0, returns 0.

        S_e(h) = (1 + (alpha|h|)^n)^(-m), m = 1 - 1/n
        dSe/dh = m*n*alpha*(alpha*x)^(n-1)*(1+(alpha*x)^n)^(-(m+1)),
                 where x=-h>0
        """
        if h >= 0.0:
            return 0.0

        x = -h  # x>0
        ax_n = (self.alpha * x) ** self.n
        factor = self.m * self.n * self.alpha * (self.alpha * x) ** (self.n - 1.0)
        se_term = (1.0 + ax_n) ** (-(self.m + 1.0))
        dSe_dh = factor * se_term  # + sign because dh/dx = -1
        return (self.theta_s - self.theta_r) * dSe_dh

    def water_content_vg_only(self, h: float) -> float:
        S_e = (1.0 + (self.alpha * np.abs(h)) ** self.n) ** (-self.m)
        return self.theta_r + (self.theta_s - self.theta_r) * S_e

    # ------------------------------------------------------------
    # 4) Function for root-finding: f(h) = capacity_vg_only(h) - storativity
    # ------------------------------------------------------------
    def _f_capacity_minus_storativity(self, h: float) -> float:
        """
        We want f(h)=0 => capacity_vg_only(h)=storativity.
        """
        return self.capacity_vg_only(h) - self.storativity

    # ------------------------------------------------------------
    # 5) h_crit: solve capacity_vg_only(h)=storativity.
    #    Use h_max_cap/2 as initial guess in a derivative-free method.
    # ------------------------------------------------------------
    @cached_property
    def h_sat(self) -> float:
        """
        Find negative h_crit that solves capacity_vg_only(h)=storativity.
        If storativity < 1e-10, return 0.0.
        If storativity <= 1e-14 or solver fails, fallback to -1e12.
        """
        if self.storativity < 1e-10:
            return 0.0

        # We'll pick h0 = h_max_cap/2 as the initial guess bounds
        h0 = self.h_max_cap / 2.0  # negative, but not too large in magnitude
        try:
            sol = root_scalar(
                self._f_capacity_minus_storativity,
                bracket=[h0, 0],  # Using [h0, 0] as the interval
                method='bisect',
                xtol=1e-12
            )
            if sol.converged:
                return sol.root
            else:
                return 0.0
        except (ValueError, RuntimeError):
            # If solver didn't converge or bounds were invalid,
            # fallback to no-solution placeholder
            return 0.0

    @property
    def th_sat(self):
        return self.water_content_vg_only(self.h_sat)

    @property
    def th_diff(self):
        return self.theta_s - self.theta_r


class SoilMaterialManager:
    """
    Manages multiple soil materials (each with its own VanGenuchtenParams)
    and a per-node material ID array that maps each node to its material.

    Demonstrates a 'switch' at h_sat:
      - If h[i] >= h_sat_vector[i], treat node i as saturated.
      - Otherwise, use van Genuchten formulas for relative saturation,
        water content, capacity, and conductivity.
    """

    def __init__(self, materials: List[VanGenuchtenParams], mat_ids: np.ndarray):
        """
        Initialize SoilMaterialManager with materials and mat_ids.
        Computes all required vectors for a given set of materials and mat_ids.

        Parameters
        ----------
        materials : list[VanGenuchtenParams]
            List of Van Genuchten parameters for each material.
        mat_ids : np.ndarray
            Array mapping each node to its material.
        """
        self.materials = materials
        self.mat_ids = mat_ids

        # Attributes to extract from VanGenuchtenParams
        self.attributes = [
            "theta_r", "theta_s", "alpha", "n", "K_s", "l", "storativity", "h_sat", "th_sat", "th_diff"
        ]

        # Initialize vectors dynamically using a loop
        for attr in self.attributes:
            setattr(
                self, attr,
                np.array([getattr(materials[m_id], attr) for m_id in mat_ids]).reshape(-1, 1)
            )

        # Compute m_vector separately as it depends on n_vector
        self.m = 1.0 - 1.0 / self.n
        self.attributes.append('m')

    def make_vector_params_(self, cols=1):
        if self.n.shape[1] == cols:
            return
        # Initialize vectors dynamically using a loop
        for attr in self.attributes:
            a_vec = getattr(self, attr)
            setattr(
                self, attr,
                np.repeat(a_vec[:, 0:1], cols, axis=1)
            )


    # ------------------------------------------------------------------
    # 1) Relative saturation: S_e(h)
    # ------------------------------------------------------------------
    def relative_saturation(self, h_in: np.ndarray) -> np.ndarray:
        """
        Piecewise:
          - If h[i] >= h_sat[i], we treat S_e=1 (fully saturated).
          - Else standard van Genuchten formula:
                S_e = [1 + (alpha*abs(h))^n]^(-m).
        """
        h = np.atleast_2d(h_in)
        self.make_vector_params_(cols=h.shape[1])

        S_e = np.zeros_like(h)
        sat_mask = h >= self.h_sat

        # Saturated region => S_e=1
        S_e[sat_mask] = 1.0

        # Unsaturated region => VG formula
        unsat_mask = ~sat_mask
        alpha_u = self.alpha[unsat_mask]
        n_u = self.n[unsat_mask]
        m_u = self.m[unsat_mask]
        h_u = h[unsat_mask]
        # S_e = (1 + (alpha*|h|)^n)^(-m)
        A = (alpha_u * np.abs(h_u)) ** n_u
        S_e[unsat_mask] = (1.0 + A) ** (-m_u)

        return S_e.reshape(h_in.shape)

    # ------------------------------------------------------------------
    # 2) Volumetric water content: theta(h)
    # ------------------------------------------------------------------
    def water_content(self, h_in: np.ndarray) -> np.ndarray:
        """
        Piecewise:
          - If h[i] >= h_sat[i], theta=theta_s  (fully saturated).
          - Else: theta=theta_r + (theta_s - theta_r)*S_e
        """
        h = h_in.reshape(len(h_in), -1)
        self.make_vector_params_(cols=h.shape[1])

        theta_out = np.zeros_like(h)
        sat_mask = (h >= self.h_sat)
        unsat_mask = ~sat_mask

        theta_out[sat_mask] = self.th_sat[sat_mask] \
                               + self.storativity[sat_mask] * (h[sat_mask] - self.h_sat[sat_mask])

        alpha_u = self.alpha[unsat_mask]
        n_u = self.n[unsat_mask]
        m_u = self.m[unsat_mask]
        theta_r_u = self.theta_r[unsat_mask]

        # Compute S_e for unsat portion:
        h_u = h[unsat_mask]
        A = (alpha_u * np.abs(h_u)) ** n_u
        S_e_u = (1.0 + A) ** (-m_u)

        theta_out[unsat_mask] = theta_r_u + self.th_diff[unsat_mask] * S_e_u
        return theta_out.reshape(h_in.shape)

    # ------------------------------------------------------------------
    # 3) Specific moisture capacity: C(h) = dtheta/dh
    # ------------------------------------------------------------------
    def capacity(self, h_in: np.ndarray) -> np.ndarray:
        """
        Piecewise:
          - If h[i] >= h_sat[i], capacity=0 (no change in theta w.r.t h).
          - Else use derivative of VG formula:
                dtheta/dh = (theta_s - theta_r)* dSe/dh,
                dSe/dh = m*n*alpha*(alpha*|h|)^(n-1)*[1+(alpha*|h|)^n]^(-(m+1)).
        """
        h = h_in.reshape(len(h_in), -1)
        self.make_vector_params_(cols=h.shape[1])

        C = np.zeros_like(h)
        sat_mask = h >= self.h_sat
        unsat_mask = ~sat_mask

        # Unsaturated region => compute derivative
        alpha_u = self.alpha[unsat_mask]
        n_u = self.n[unsat_mask]
        m_u = self.m[unsat_mask]
        h_u = h[unsat_mask]

        x = np.abs(h_u)
        ax_n = (alpha_u * x) ** n_u
        factor = m_u * n_u * alpha_u * (alpha_u * x) ** (n_u - 1.0)
        dSe_dh = factor * (1.0 + ax_n) ** (-(m_u + 1.0))

        C[unsat_mask] = self.th_diff[unsat_mask] * dSe_dh
        C[sat_mask] = self.storativity[sat_mask]
        return C.reshape(h_in.shape)

    # ------------------------------------------------------------------
    # 4) Hydraulic conductivity: K(h)
    # ------------------------------------------------------------------
    def hydraulic_conductivity(self, h_in: np.ndarray) -> np.ndarray:
        """
        Piecewise:
          - If h[i] >= h_sat[i], K=K_s (fully saturated).
          - Else use Mualem-van Genuchten:
                K(h) = K_s * S_e^l * [1 - (1 - S_e^(1/m))^m]^2
        """
        h = h_in.reshape(len(h_in), -1)
        self.make_vector_params_(cols=h.shape[1])

        K_out = np.zeros_like(h)
        sat_mask = h >= 0.0
        unsat_mask = ~sat_mask

        # Saturated => K=K_s
        K_out[sat_mask] = self.K_s[sat_mask]

        # Unsaturated => Mualem-van Genuchten formula
        l_u = self.l[unsat_mask]
        K_s_u = self.K_s[unsat_mask]
        alpha_u = self.alpha[unsat_mask]
        n_u = self.n[unsat_mask]
        m_u = self.m[unsat_mask]
        h_u = h[unsat_mask]

        A = (alpha_u * np.abs(h_u)) ** n_u
        S_e_u = (1.0 + A) ** (-m_u)

        S_e_pow_l = S_e_u ** l_u
        Se_1m = S_e_u ** (1.0 / m_u)
        bracket = 1.0 - (1.0 - Se_1m)**(m_u)
        K_out[unsat_mask] = K_s_u * S_e_pow_l * (bracket**2)

        return K_out.reshape(h_in.shape)

    # ------------------------------------------------------------------
    # 5) Inverse function: from water content -> pressure head
    # ------------------------------------------------------------------
    def pressure_head_from_theta(self, th_in: np.ndarray) -> np.ndarray:
        """
        Piecewise inversion:
          - If theta >= theta_s => h = h_sat (fully saturated).
          - Else standard VG inversion for 0 < Se < 1
                h = - 1/alpha * [Se^(-1/m) - 1]^(1/n),
            or large negative if Se <= 0.
        """
        theta = th_in.reshape(len(th_in), -1)
        self.make_vector_params_(cols=theta.shape[1])

        h_out = np.zeros_like(theta)
        # 1) If theta >= theta_s => saturate => h = h_sat
        sat_mask = (theta >= self.th_sat)
        h_out[sat_mask] = (theta[sat_mask] - self.th_sat[sat_mask]) / self.storativity[sat_mask] \
                          + self.h_sat[sat_mask]

        # 2) Unsaturated but Se>0 => standard VG
        unsat_mask = ~sat_mask
        Se_u = (theta[unsat_mask] - self.theta_r[unsat_mask]) / self.th_diff[unsat_mask]
        a_u  = self.alpha[unsat_mask]
        n_u  = self.n[unsat_mask]
        m_u  = self.m[unsat_mask]
        h_out[unsat_mask] = -1.0 / a_u * (Se_u ** (-1.0 / m_u) - 1.0) ** (1.0 / n_u)

        return h_out.reshape(th_in.shape)

def plot_soils(soils: List[VanGenuchtenParams], fname=None):
    n_mat = len(soils)
    # Array of pressure heads (from -10 up to +1, let's say)
    h_values = np.linspace(-50, 1, 500)  # length e.g. 300
    H_all = np.ones(n_mat)[:, None] * h_values[None, :]

    # 2) We want to build a single big array of heads, H_all,
    #    which concatenates the h_values block for each material.
    #    For example, if we have 5 materials and len(h_values)=300,
    #    then len(H_all)=5*300=1500.


    # 3) Build mat_ids so that each block of 300 belongs to one material
    mat_ids = np.arange(n_mat, dtype=int) #[:, None] * np.ones_like(h_values,dtype=int)[None, :]

    # 4) Create the manager with multiple materials
    manager = SoilMaterialManager(materials=soils, mat_ids=mat_ids)

    mat_shape = (n_mat, len(h_values))
    #H_all = H_all.ravel()
    # 5) Compute water_content, conductivity, capacity for entire H_all
    Theta_all = manager.water_content(H_all)    #.reshape(*mat_shape)
    K_all     = manager.hydraulic_conductivity(H_all)   #.reshape(*mat_shape)
    C_all     = manager.capacity(H_all) #.reshape(*mat_shape)


    diff = manager.pressure_head_from_theta(Theta_all) - H_all
    assert np.allclose(manager.pressure_head_from_theta(Theta_all), H_all)
    eps = 1.0e-6
    th_1 = manager.water_content(H_all + eps)
    C_approx = (th_1 - Theta_all) / eps
    C_diff = C_all - C_approx
    assert np.allclose(C_all, C_approx)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, n_mat))

    for i, (soil, color) in enumerate(zip(soils, colors)):
        label_ = f"n={soil.n}, $\\alpha$={soil.alpha}"
        axs[0].plot(Theta_all[i], h_values, color=color, label=label_)
        axs[1].plot(K_all[i],     h_values, color=color, label=label_)
        axs[2].plot(C_all[i],     h_values, color=color, label=label_)

    axs[0].set_title("Water Content (θ) vs. Head (h)")
    axs[0].set_xlabel("θ")
    axs[0].set_ylabel("Pressure head h [L]")
    axs[0].legend()

    axs[1].set_title("Hydraulic Conductivity (K) vs. Head (h)")
    axs[1].set_xlabel("K")

    axs[2].set_title("Specific Capacity (C) vs. Head (h)")
    axs[2].set_xlabel("C")

    for ax in axs:
        ax.invert_xaxis()
        ax.grid(True)
        ax.set_xscale('log')

    fig.tight_layout()
    if fname is None:
        plt.show()
    else:
        fig.savefig(fname)

def plotting_test_manager():
    """
    In this version, we create multiple materials (one for each n-value),
    then replicate h_values for each material in a single manager call.
    Finally, we reshape results to plot them individually.
    """
    # Shared parameters (besides n)
    base = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.32,
        alpha=0.145,
        n=2.0,
        K_s=1.0,
        l=0.5,
        storativity=1e-6
    )
    # We'll test multiple n values
    n_values = [1.2, 1.5, 2.0, 3.0, 4.0]
    materials = [base.update(n=n) for n in n_values]
    plot_soils(materials)


if __name__ == "__main__":
    plotting_test_manager()