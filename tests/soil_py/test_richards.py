import numpy as np
import matplotlib.pyplot as plt
from richards import RichardsEquationSolver, RichardsSolverOutput
from soil import VanGenuchtenParams, plot_soils
from bc_models import dirichlet_bc, neumann_bc, free_drainage_bc, seepage_bc
from plots import plot_richards_output
import pandas as pd
import pytest
from scipy.interpolate import RegularGridInterpolator

def read_ref_sol(file_path, target_timesteps, target_z_nodes):
    """
    Reads a CSV file containing reference solutions for the Richards equation and interpolates
    it to the specified timesteps and spatial nodes using 2D interpolation.

    Parameters:
    - file_path (str): Path to the CSV file.
    - target_timesteps (list or np.ndarray): Timesteps to interpolate to.
    - target_z_nodes (list or np.ndarray): Spatial node positions to interpolate to.

    Returns:
    - dict: A dictionary containing interpolated pressure values.
      {
        "timesteps": np.ndarray,  # Target timesteps
        "z_nodes": np.ndarray,    # Target spatial nodes
        "pressure": np.ndarray    # Interpolated pressures (len(timesteps), len(z_nodes))
      }
    """
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Extract time column and spatial columns
    time_column = data.columns[0]
    spatial_columns = data.columns[1:]  # All columns except the time column

    # Convert spatial columns (z=0.00, z=0.01, ...) to numeric values
    z_positions = np.array([float(col.split('=')[1]) for col in spatial_columns])
    z_positions = z_positions - np.max(z_positions)

    # Convert the DataFrame to NumPy arrays for fast processing
    time_values = data[time_column].values  # Timesteps
    pressure_values = data[spatial_columns].values  # Shape: (num_times, num_z_positions)

    # Create a 2D interpolator
    interpolator = RegularGridInterpolator(
        (time_values, z_positions), pressure_values, method="linear", bounds_error=False, fill_value=None
    )

    # Create a meshgrid of target timesteps and target z_nodes for interpolation
    target_mesh = np.array(np.meshgrid(target_timesteps, target_z_nodes, indexing="ij")).reshape(2, -1).T

    # Interpolate using the 2D interpolator
    interpolated_pressures = interpolator(target_mesh).reshape(len(target_timesteps), len(target_z_nodes))

    return interpolated_pressures

# Example usage:
# file_path = "reference_solution.csv"
# target_timesteps = [0, 1, 2, 3, 4]  # Replace with actual desired timesteps
# target_z_nodes = [0.1, 0.2, 0.3]    # Replace with actual desired z-node positions
# result = read_and_interpolate_ref_solution_2d(file_path, target_timesteps, target_z_nodes)
# print(result)

def test_to_parflow():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    Key ParFlow setting:

    Z step: 1cm, 200 steps
    T step: 0.01, end time 24
    Tolerances:
    self._run.Solver.MaxIter = 25000
    self._run.Solver.AbsTol = 1e-12
    self._run.Solver.Drop = 1e-20

    self._run.Solver.Nonlinear.MaxIter = 300
    self._run.Solver.Nonlinear.ResidualTol = 1e-6
    self._run.Solver.Nonlinear.StepTol = 1e-30
    self._run.Solver.Nonlinear.EtaChoice = "EtaConstant"
    self._run.Solver.Nonlinear.Globalization = "LineSearch"
    self._run.Solver.Nonlinear.EtaValue = 1e-3
    self._run.Solver.Nonlinear.UseJacobian = True
    self._run.Solver.Nonlinear.DerivativeEpsilon = 1e-12

    self._run.Solver.Linear.KrylovDimension = 20
    self._run.Solver.Linear.MaxRestart = 2
    self._run.Solver.Linear.Preconditioner = "MGSemi"
    self._run.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
    self._run.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10


    """

    # Common parameters
    n_nodes = 200
    # z_bottom = -2.0  # [m]
    z_bottom = -2.0  # [m]
    t_span = (0, 24)  # [h]Simulate for 1 hour
    t_out = 0.01  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params_JB = VanGenuchtenParams(
        theta_r=0.045,  # 0.06
        theta_s=0.43,  # 0.47
        alpha=0.15,  # 0.58
        n=1.56,  # 3.7
        K_s=0.01,  # 1.2833e-2
        storativity=1e-4
    )

    vg_params_JS = VanGenuchtenParams(
        theta_r=0.06,
        theta_s=0.47,
        alpha=0.58,
        n=3.7,
        K_s=30.8 / 100 / 24 * 100,        # =1.2833e-2  [m/h] ; time scale given implicitely
        storativity=1e-4
    )
    vg_params = vg_params_JS
    plot_soils([vg_params_JS, vg_params_JB], "soil_params_dirichlet_test.pdf")

    # Initial conditions: linear pressure gradient, top to bot
    # h_initial = np.linspace(-20, -100, n_nodes)
    h_initial = np.linspace(-20, -100, n_nodes)

    # Test 1: Seepage on top, Dirichlet at bottom
    print("Running Test 1: Seepage on top, Dirichlet at bottom")
    dirichlet_top_bc = dirichlet_bc(h_target=-0.01)
    # dirichlet_bottom_bc = dirichlet_bc(h_target=-100.0)
    dirichlet_bottom_bc = dirichlet_bc(h_target=-100.0)

    solver_1 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (dirichlet_top_bc, dirichlet_bottom_bc)
    )

    result = solver_1.richards_solver(h_initial, t_span, t_out) #, method='LSODA')
    ref_sol = read_ref_sol('parflow_reference/pressure-dirichlet.csv',
                           result.times, result.nodes_z)

    ref_res = solver_1.make_result(ref_sol.T, result.times, result.nodes_z)
    plot_richards_output(result, fname="richards_result.pdf")
    plot_richards_output(ref_res, fname="ref_result.pdf")
    plt.show()

    h_diff = np.abs(result.H - ref_sol["pressure"]).max()
    print("max diff", h_diff)
    assert h_diff < 1e-3

@pytest.mark.skip
def test_bc_dirichlet():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    """

    # Common parameters
    n_nodes = 50
    #z_bottom = -2.0  # [m]
    z_bottom = -0.2  # [m]
    t_span = (0, 3600)  # Simulate for 1 hour
    t_out = 60  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params_JB = VanGenuchtenParams(
        theta_r=0.045,       # 0.06
        theta_s=0.43,        # 0.47
        alpha=0.15,          # 0.58
        n=1.56,              # 3.7
        K_s=0.01,            # 1.2833e-2
        storativity=1e-4
    )

    vg_params_JS = VanGenuchtenParams(
        theta_r=0.06,
        theta_s=0.47,
        alpha=0.58,
        n=3.7,
        K_s=1.2833e-2,      # m/h ; time scale given implicitely
        storativity=1e-4
    )
    vg_params = vg_params_JS
    plot_soils([vg_params_JS, vg_params_JB], "soil_params_dirichlet_test.pdf")

    # Initial conditions: linear pressure gradient, top to bot
    #h_initial = np.linspace(-20, -100, n_nodes)
    h_initial = np.linspace(-20, -30, n_nodes)

    # Test 1: Seepage on top, Dirichlet at bottom
    print("Running Test 1: Seepage on top, Dirichlet at bottom")
    dirichlet_top_bc = dirichlet_bc(h_target=-0.01)
    #dirichlet_bottom_bc = dirichlet_bc(h_target=-100.0)
    dirichlet_bottom_bc = dirichlet_bc(h_target=-30.0)

    solver_1 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (dirichlet_top_bc, dirichlet_bottom_bc)
    )

    result = solver_1.richards_solver(h_initial, t_span, t_out, method='LSODA')

    plot_richards_output(result)
    plt.show()


@pytest.mark.skip
def test_bc_seepage_dirichlet():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    """

    # Common parameters
    n_nodes = 50
    z_bottom = -2.0  # [m]
    t_span = (0, 3600)  # Simulate for 1 hour
    t_out = 300  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        storativity=1e-4
    )

    # Initial conditions: linear pressure gradient
    h_initial = np.linspace(-5, -1, n_nodes)

    # Test 1: Seepage on top, Dirichlet at bottom
    print("Running Test 1: Seepage on top, Dirichlet at bottom")
    seepage_top_bc = seepage_bc(q_given=0.001, h_crit=0.0, transition_width=0.5)
    dirichlet_bottom_bc = dirichlet_bc(h_target=-1.0)

    solver_1 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (seepage_top_bc, dirichlet_bottom_bc)
    )

    result = solver_1.richards_solver(h_initial, t_span, t_out, method='LSODA')

    plot_richards_output(result)
    plt.show()


@pytest.mark.skip
def test_bc_drainage_neumann():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    """

    # Common parameters
    n_nodes = 50
    z_bottom = -2.0  # [m]
    t_span = (0, 3600)  # Simulate for 1 hour
    t_out = 300  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        l=0.5
    )

    # Initial conditions: linear pressure gradient
    h_initial = np.linspace(-5, -1, n_nodes)

    # Test 2: Neumann (inflow) on top, Free drainage at bottom
    print("Running Test 2: Neumann (inflow) on top, Free drainage at bottom")
    inflow_top_bc = neumann_bc(q=0.002)
    free_drainage_bottom_bc = free_drainage_bc()

    solver_2 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (inflow_top_bc, free_drainage_bottom_bc)
    )

    times, results = solver_2.richards_solver(h_initial, t_span, t_out)

    # Plot results for Test 2
    plt.figure()
    for i, h in enumerate(results.T):
        plt.plot(np.linspace(0, z_bottom, n_nodes), h, label=f"t={times[i]:.0f}s")
    plt.title("Test 2: Inflow on top, Free drainage at bottom")
    plt.xlabel("Depth (m)")
    plt.ylabel("Pressure Head (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
#     test_bc_dirichlet()
#     #test_bc_seepage_dirichlet()
#     #test_bc_drainage_neumann()
