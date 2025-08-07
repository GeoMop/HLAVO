import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
from parflow_model import ToyProblem
from soil_model.plots import RichardsSolverOutput, plot_richards_output

if __name__ == "__main__":
    # An argument parser to handle command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Path to work dir')  # First argument: working directory path
    parser.add_argument('model_config_file', help='Path to configuration file')  # Second argument: model configuration file path
    args = parser.parse_args(sys.argv[1:])  # Parse the arguments provided via command line (excluding the script name)

    # Load model configuration from the YAML file
    with Path(args.model_config_file).open("r") as f:
        model_config = yaml.safe_load(f)

    # Convert the work directory string to a Path object
    work_dir_path = Path(args.work_dir)

    # Initialize the model with configuration and output directory
    model = ToyProblem(model_config, workdir=work_dir_path / "output-toy")

    # Generate an initial linear pressure field based on the model configuration
    pressure = model.make_linear_pressure(model_config)

    # Define simulation parameters
    precipitation_flux = -0.0166  # Constant precipitation input (negative indicates infiltration)
    stop_time = 200  # Duration of each simulation period
    model_time_step = 0.025  # Time step used for the simulation
    n_time_periods = 5  # Number of sequential simulation periods

    # Initialize lists to store time and simulation outputs
    times = []
    pressure_list = []
    moisture_list = []

    # Loop through each time period and run the model
    for i in range(n_time_periods):
        # Run the model with the given pressure, precipitation, and timing parameters
        model.run(init_pressure=pressure, precipitation_value=precipitation_flux,
                  start_time=0, stop_time=stop_time, time_step=model_time_step)

        # Retrieve the pressure and moisture results at the end of this period
        pressure = model.get_data(current_time_step=stop_time, data_name="pressure")
        moisture = model.get_data(current_time_step=stop_time, data_name="moisture")

        # Record the current simulation time and results
        times.append((i + 1) * stop_time)
        pressure_list.append(pressure)
        moisture_list.append(moisture)

    # Get the z-coordinates of the model nodes (e.g., for plotting vs. depth)
    nodes_z = model.get_nodes_z()

    # Wrap the results into a structured output object
    output = RichardsSolverOutput(
        np.array(times),                   # Times at which data was recorded
        np.array(pressure_list).T,         # Transpose pressure to shape (n_nodes, n_times)
        np.array(moisture_list),           # Moisture content
        None, None,                        # Placeholders for optional data (e.g., fluxes)
        nodes_z                            # Node depth/elevation
    )

    # Generate and show a plot of the results, and save it as a PDF
    plot_richards_output(output, [], work_dir_path / "mean_solution.pdf", show=True)
