import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
from hlavo.soil_parflow.parflow_model import ToyProblem
from hlavo.kalman.visualization.plots import RichardsSolverOutput, plot_richards_output


def run_parflow_model(tmp_path: Path):
    """
    Unit test for ToyProblem Richards solver.
    Runs a short transient simulation and checks
    output consistency.
    """

    # ------------------------------------------------------------------
    # Embedded model configuration (formerly YAML)
    # ------------------------------------------------------------------
    model_config = {
        "init_pressure": [-40, -100],  # bottom, top
        "static_params": {
            "ComputationalGrid": {
                "Lower.Z": -135,
                "DZ": 9,
                "NZ": 15,
            },
            "BCPressure_bottom": 0,
        },
        "params": {
            "vG_K_s": "Geom.domain.Perm.Value",
            "vG_n": [
                "Geom.domain.Saturation.N",
                "Geom.domain.RelPerm.N",
            ],
            "vG_Th_s": "Geom.domain.Saturation.SSat",
            "vG_Th_r": "Geom.domain.Saturation.SRes",
            "vG_alpha": [
                "Geom.domain.Saturation.Alpha",
                "Geom.domain.RelPerm.Alpha",
            ],
        },
    }

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------
    work_dir = tmp_path / "output-toy"
    model = ToyProblem(model_config, workdir=work_dir)

    pressure = model.make_linear_pressure(model_config)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    precipitation_flux = -0.0166
    stop_time = 200
    model_time_step = 0.025
    n_time_periods = 5

    times = []
    pressure_list = []
    moisture_list = []

    # ------------------------------------------------------------------
    # Time stepping loop
    # ------------------------------------------------------------------
    for i in range(n_time_periods):
        model.run(
            init_pressure=pressure,
            precipitation_value=precipitation_flux,
            start_time=0,
            stop_time=stop_time,
            time_step=model_time_step,
        )

        pressure = model.get_data(
            current_time_step=stop_time,
            data_name="pressure",
        )
        moisture = model.get_data(
            current_time_step=stop_time,
            data_name="moisture",
        )

        times.append((i + 1) * stop_time)
        pressure_list.append(pressure)
        moisture_list.append(moisture)

    nodes_z = model.get_nodes_z()
    el_centers = model.get_el_centers_z()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    pressure_list_arr = np.array(pressure_list)
    moisture_list_arr = np.array(moisture_list)
    assert len(times) == n_time_periods
    assert pressure_list_arr.shape[0] == n_time_periods
    assert pressure_list_arr.shape[1] == len(el_centers)
    assert moisture_list_arr.shape[0] == n_time_periods

    # Physical sanity checks
    assert np.all(np.isfinite(pressure_list_arr))
    assert np.all(np.isfinite(moisture_list_arr))
    assert np.all(moisture_list_arr >= 0.0)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    output = RichardsSolverOutput(
        np.array(times),
        np.array(pressure_list).T,
        np.array(moisture_list),
        None,
        None,
        nodes_z,
    )

    # Generate and show a plot of the results, and save it as a PDF
    #plot_richards_output(output, [], work_dir / "mean_solution.pdf", show=True)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        run_parflow_model(Path(tmp))

