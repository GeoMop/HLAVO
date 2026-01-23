import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from soil import VanGenuchtenParams, SoilMaterialManager
import attrs

@attrs.define
class RichardsSolverOutput:
    """
    Output class for Richards Equation Solver.

    Attributes:
    ----------
    times : np.ndarray
        Array of output times.
    H : np.ndarray, times in rows.
        Pressure head array at output times.
    moisture : np.ndarray
        Moisture content (absolute saturation) at output times.
    top_flux : np.ndarray
        Influx at the top boundary at output times.
    bottom_flux : np.ndarray
        Influx at the bottom boundary at output times.
    """
    times: np.ndarray
    H: np.ndarray
    moisture: np.ndarray
    top_flux: np.ndarray
    bottom_flux: np.ndarray
    nodes_z: np.ndarray

class RichardsEquationSolver:
    def __init__(self, nodes_z, vg_params: VanGenuchtenParams, bc_funcs):
        """
        Initialize the RichardsEquationSolver for a given spatial discretization.

        Parameters:
        nodes_z : np.ndarray
            Elevations of the nodes in the spatial domain.
        soil_manager : SoilMaterialManager
            Object to manage soil properties.
        bc_fun : pair of callables (bc_top, bc_bottom), callable(t, H, K, dz)
            H - top/bot pressure head
            K - top/bot hydraulic conductivity
            dz - top/bot mesh step
            return top/bot water inflow
        """
        self.nodes_z = nodes_z
        self.soil_manager = SoilMaterialManager([vg_params], np.zeros_like(nodes_z, dtype=int))
        self.bc = bc_funcs

    @staticmethod
    def from_uniform_mesh(n_nodes, z_bot, soil_manager, bc_funcs):
        """
        Create a RichardsEquationSolver instance for a uniform mesh.

        Parameters:
        n_nodes : int
            Number of nodes in the spatial domain.
        dz : float
            Constant spacing between nodes.
        z_bot : float
            Bottom elevation of the domain.
        soil_manager : SoilMaterialManager
            Object to manage soil properties.
        q_top : callable
            Function for top boundary flux.
        q_bot : callable
            Function for bottom boundary flux.

        Returns:
        RichardsEquationSolver
            Configured instance of the solver.
        """
        nodes_z = np.linspace(0, z_bot, n_nodes)
        return RichardsEquationSolver(nodes_z, soil_manager, bc_funcs)


    def optimal_nodes(self, z_existing, h_existing, K_existing, z_bottom, z_top, epsilon=1e-6, max_nodes=1000):
        """
        Computes the z-coordinates of nodes in a 1D mesh using an adaptive step size formula,
        taking precomputed h and K as vectors at existing nodes.

        Parameters:
            z_existing (np.array): Array of existing z-coordinates of the domain.
            h_existing (np.array): Array of pressure head values corresponding to z_existing.
            K_existing (np.array): Array of hydraulic conductivity values corresponding to z_existing.
            z_bottom (float): The starting z-coordinate of the domain (bottom).
            z_top (float): The ending z-coordinate of the domain (top).
            epsilon (float): A small constant to avoid division by zero.
            max_nodes (int): Maximum number of nodes to generate.

        Returns:
            np.array: Array of new z-coordinates for the nodes.
        """
        # Initialize the node list with the bottom boundary
        z_nodes = [z_bottom]
        z_current = z_bottom

        # Interpolators for h and K
        h_interp = lambda z: np.interp(z, z_existing, h_existing)
        K_interp = lambda z: np.interp(z, z_existing, K_existing)

        while z_current < z_top and len(z_nodes) < max_nodes:
            # Compute the interpolated pressure head and its gradient
            h_current = h_interp(z_current)
            h_next = h_interp(z_current + epsilon)
            h_gradient = (h_next - h_current) / epsilon  # Numerical gradient

            # Compute the interpolated hydraulic conductivity
            K_current = K_interp(z_current)

            # Compute the step size formula
            step_size = (
                    K_current * min(1, abs(h_gradient + 1)) /
                    (abs(h_gradient) + epsilon)
            )

            # Update the current z-coordinate
            z_current += step_size

            # Avoid overshooting the top boundary
            if z_current > z_top:
                z_current = z_top

            # Add the new node to the list
            z_nodes.append(z_current)

        return np.array(z_nodes)

    def plot_iter(self, t, H):
        t_iter = self._iters.setdafault(t, 0)
        import matplotlib.pyplot as plt
        plt.plot(self.nodes_z, H)
        plt.xlabel(f"t={t}, iter={t_iter}, Z position")
        plt.ylabel(f"h")
        self._iters[t] = t_iter + 1
        plt.show()


    def compute_rhs(self, t, H):
        """
        Assemble the right-hand side of the 1D Richards equation ODE
        in standard form, using a lumped (diagonal) mass matrix.

        Parameters:
        t : float
            Current time.
        h : np.ndarray
            Pressure head distribution at the current time step.

        Returns:
        np.ndarray
            Time derivative of pressure head (dH/dt).
        """
        # Number of elements (N = number of intervals)
        d = np.abs(np.diff(self.nodes_z))  # element lengths, shape (N,)
        Cvals = self.soil_manager.capacity(H)  # C at each node, shape (N+1,)

        M_lump = np.zeros_like(H)
        M_lump[0] = 0.5 * d[0] * Cvals[0]
        M_lump[1:-1] = 0.5 * (d[:-1] + d[1:]) * Cvals[1:-1]
        M_lump[-1] = 0.5 * d[-1] * Cvals[-1]

        Kvals = self.soil_manager.hydraulic_conductivity(H)  # K at each node, shape (N+1,)
        numerator = 2.0 * Kvals[:-1] * Kvals[1:]
        denominator = (Kvals[:-1] + Kvals[1:])
        K_e = numerator / denominator  # shape (N,)

        dHdZ = (H[1:] - H[:-1]) / d  # partial derivative on each element, shape (N,)
        flux = K_e * (dHdZ + 1.0)  # shape (N,)

        res = np.zeros_like(H)
        res[:-1] += flux
        res[1:] -= flux

        bc_z = [0, -1]
        bc_top_fn, bc_bot_fn = self.bc
        res[0] += bc_top_fn(t, H[0], Kvals[0], d[0])
        res[-1] -= bc_bot_fn(t,  H[-1], Kvals[-1], d[-1])

        # Weak Dirichlet
        # bottom_flux_correction = Kvals[-1] * (H[-1] - h_bot) / d[-1]
        # b_vec[-1] -= bottom_flux_correction

        dHdt = res / M_lump  # elementwise division since M_lump is diagonal

        return dHdt

    def compute_jacobian(self, t, h):
        """
        Compute the Jacobian matrix for the ODE system using finite differences.

        Parameters:
        t : float
            Current time.
        h : np.ndarray
            Pressure head distribution at the current time step.

        Returns:
        np.ndarray
            Jacobian matrix of the system.
        """
        n_nodes = len(h)
        epsilon = 1e-6
        jacobian = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            h_perturbed = h.copy()
            h_perturbed[i] += epsilon
            rhs_perturbed = self.compute_rhs(t, h_perturbed)
            rhs_original = self.compute_rhs(t, h)
            jacobian[:, i] = (rhs_perturbed - rhs_original) / epsilon

        return jacobian

    # def richards_solver(self, h_initial, t_span, t_out, method='LSODA'):
    #     """
    #     Solve Richards' equation using `solve_ivp` and save results at specified output times.
    #
    #     Parameters:
    #     h_initial : np.ndarray
    #         Initial pressure head distribution (1D array).
    #     t_span : tuple
    #         Start and end time for the simulation.
    #     t_out : float or list
    #         If float, defines the interval between output times.
    #         If list, specifies explicit output times.
    #     method : str
    #         Solver method, default is 'Radau' (suitable for stiff problems).
    #
    #     Returns:
    #     List[Tuple[float, np.ndarray]]
    #         List of tuples containing time and pressure head distribution at each output time.
    #     """
    #     # Determine output times
    #     if isinstance(t_out, (int, float)):
    #         output_times = np.arange(t_span[0], t_span[1] + t_out, t_out)
    #     elif isinstance(t_out, list):
    #         output_times = np.array(t_out)
    #     else:
    #         raise ValueError("t_out must be a float or a list of times.")
    #
    #     # Solve using solve_ivp
    #     solution = solve_ivp(
    #         fun=self.compute_rhs,
    #         t_span=t_span,
    #         y0=h_initial,
    #         method=method,
    #         jac=self.compute_jacobian,
    #         t_eval=output_times,
    #         rtol = 1e-4,
    #         atol = 1e-6
    #     )
    #
    #     # Collect results as (time, H)
    #     return output_times, solution.y

    def richards_solver(self, h_initial, t_span, t_out, method='Radau', rtol=1.0e-6, atol=1.0e-8) -> RichardsSolverOutput: #method='LSODA'):
        """
        Solve Richards' equation using `solve_ivp` and save results at specified output times.

        Parameters:
        h_initial : np.ndarray
            Initial pressure head distribution (1D array).
        t_span : tuple
            Start and end time for the simulation.
        t_out : float or list
            If float, defines the interval between output times.
            If list, specifies explicit output times.
        method : str
            Solver method, default is 'Radau' (suitable for stiff problems).

        Returns:
        RichardsSolverOutput
            Outputs of the solver including times, pressure head, moisture content, and boundary fluxes.
        """
        # Determine output times
        if isinstance(t_out, (int, float)):
            output_times = np.arange(t_span[0], t_span[1] + t_out, t_out)
        elif isinstance(t_out, list):
            output_times = np.array(t_out)
        else:
            raise ValueError("t_out must be a float or a list of times.")

        # Solve using solve_ivp
        solution = solve_ivp(
            fun=self.compute_rhs,
            t_span=t_span,
            y0=h_initial,
            method=method,
            jac=self.compute_jacobian,
            t_eval=output_times,
            rtol=rtol,
            atol=atol
        )
        if not solution.success:
            raise Exception(f"Richards ODE solver failed after {len(solution.t)} time steps.\n" + f"Message: {solution.message}")
        H = (solution.y)  # time points at rows
        return self.make_result(H, output_times, self.nodes_z)


    def make_result(self, H: np.ndarray, output_times, nodes_z) -> RichardsSolverOutput:
        """
        Create a RichardsSolverOutput object from the solution of the ODE solver.
        :param h:
        :return:
        """
        moisture = self.soil_manager.water_content(H).T

        d = np.diff(nodes_z)
        Kvals = self.soil_manager.hydraulic_conductivity(H).T
        H = H.T
        bc_top_fn, bc_bot_fn = self.bc
        top_flux = bc_top_fn(output_times, H[:, 0], Kvals[:, 0], d[0])
        bottom_flux = bc_bot_fn(output_times, H[:, -1], Kvals[:, -1], d[-1])

        return RichardsSolverOutput(
            times=output_times,
            H=H,                  # (n_times, n_nodes)
            moisture=moisture,      # (n_times, n_nodes)
            top_flux=top_flux,
            bottom_flux=bottom_flux,
            nodes_z=nodes_z
        )


def test_richards_solver():
    """
    Test the RichardsEquationSolver with a simple setup.
    """
    from soil import VanGenuchtenParams, SoilMaterialManager

    # Define Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        l=0.5
    )

    # Define boundary conditions and initial state
    n_nodes = 10
    z_bottom = 1.3  # [m]
    h_initial = np.linspace(-5, -10, n_nodes)  # Linear pressure gradient
    boundary_conditions = (-10, 0)  # Dirichlet boundary conditions
    flux_bc = (0, 0.2)  # Dirichlet boundary conditions
    t_span = (0, 1)  # Simulate from t=0 to t=1

    # Create the solver instance
    solver = RichardsEquationSolver.from_uniform_mesh(
        n_nodes,
        z_bottom,
        vg_params,
        *flux_bc
    )

    # Solve the problem
    solution = solver.richards_solver(h_initial=h_initial, t_span=t_span, t_out=0.1, method='Radau')

    # Print results
    print("Solution:", solution)


def test_jacobian_finite_differences():
    """
    Test the Jacobian computation using finite differences.
    """
    from soil import VanGenuchtenParams, SoilMaterialManager

    # Define Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        l=0.5
    )

    # Define boundary conditions and initial state
    z_bottom = 1.3  # [m]
    flux_bc = (0, -0.2)  # Dirichlet boundary conditions
    n_nodes = 10
    dz = 0.1
    h_initial = np.linspace(-10, -5, n_nodes)  # Linear pressure gradient
    boundary_conditions = (-10, -5)  # Dirichlet boundary conditions

    # Create the solver instance
    solver = RichardsEquationSolver.from_uniform_mesh(
        n_nodes,
        z_bottom,
        vg_params,
        *flux_bc
    )

    # Compute Jacobian at initial condition
    t = 0
    analytical_jacobian = solver.compute_jacobian(t, h_initial)

    # Compute finite difference Jacobian
    epsilon = 1e-6
    n_nodes = len(h_initial)
    finite_difference_jacobian = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        h_perturbed = h_initial.copy()
        h_perturbed[i] += epsilon
        rhs_perturbed = solver.compute_rhs(t, h_perturbed)
        rhs_original = solver.compute_rhs(t, h_initial)
        finite_difference_jacobian[:, i] = (rhs_perturbed - rhs_original) / epsilon

    # Compare analytical and finite difference Jacobians
    difference = np.linalg.norm(analytical_jacobian - finite_difference_jacobian, ord=np.inf)
    print("Jacobian difference (inf norm):", difference)
    assert difference < 1e-5, "Jacobian test failed!"


if __name__ == "__main__":
    # Run the test
    test_richards_solver()
    test_jacobian_finite_differences()
