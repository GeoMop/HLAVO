import os
import sys
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from hlavo.soil_parflow.parflow_model import ToyProblem, redirect_process_output


TEST_DIR = Path(__file__).resolve().parent


model_config = {
    "init_pressure": [-100, -40],  # top, bottom
    "static_params": {
        "ComputationalGrid": {
            "Lower.Z": -135,
            "DZ": 0.9,
            "NZ": 150,
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


clm_model_config = {
    **model_config,
    "use_clm": True,
    "clm_files": {
        "drv_vegm_file": str(TEST_DIR / "drv_vegm.dat"),
        "drv_vegp_file": str(TEST_DIR / "drv_vegp.dat"),
    }
}


def test_redirect_process_output(tmp_path):
    log_path = tmp_path / "parflow.log"
    with redirect_process_output(log_path):
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"fd stdout\n")
        os.write(2, b"fd stderr\n")

    log_text = log_path.read_text()
    assert "python stdout" in log_text
    assert "python stderr" in log_text
    assert "fd stdout" in log_text
    assert "fd stderr" in log_text


def assert_parflow_log(working_dir):
    log_path = working_dir / "parflow.log"
    assert log_path.is_file()
    log_text = log_path.read_text()
    assert "ParFlow ran successfully" in log_text


def assert_kalman_model_outputs(toy, current_time_step):
    nz = toy._run.ComputationalGrid.NZ
    expected_shapes = {
        "pressure": (nz,),
        "moisture": (nz,),
        "velocity": (nz + 1,),
    }
    for data_name, expected_shape in expected_shapes.items():
        data = toy.get_data(current_time_step=current_time_step, data_name=data_name)
        assert data.shape == expected_shape
        assert np.all(np.isfinite(data))


def test_parflow_model():

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Model initialization
        output_path = Path(tmp_dir)
        working_dir = output_path / "output-toy"
        toy = ToyProblem(model_config, workdir=working_dir)

        # Model parameters
        precipitation_flux = -0.0166
        stop_time = 1
        model_time_step = 0.5
        pressure = toy.make_linear_pressure(model_config)
        state_params = {
            "vG_K_s": 0.0737,
            "vG_n": 1.89,
            "vG_Th_s": 0.41,
            "vG_Th_r": 0.065,
            "vG_alpha": 0.75,
        }

        # Run and postprocess
        toy.run(init_pressure=pressure,
                precipitation_value=precipitation_flux,
                state_params=state_params,
                start_time=0,
                stop_time=stop_time,
                time_step=model_time_step,
                working_dir=str(working_dir))

        assert_parflow_log(working_dir)
        assert_kalman_model_outputs(toy, current_time_step=stop_time)
        toy.save_pressure(output_path / "pressure.png")


def test_parflow_model_with_clm():

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Model initialization
        output_path = Path(tmp_dir)
        working_dir = output_path / "output-toy"
        toy = ToyProblem(clm_model_config, workdir=working_dir)

        # Model parameters
        precipitation_flux = 0
        stop_time = 1.0
        model_time_step = 0.5
        pressure = toy.make_linear_pressure(model_config)

        # Define a sample xarray
        time_step = pd.Timedelta(hours=model_time_step)
        start_date = "2026-03-01"
        time = pd.date_range(start=start_date, end=pd.Timestamp(start_date)+pd.Timedelta(hours=stop_time), freq=time_step)
        ds = xr.Dataset(
            data_vars=dict(
                DSWR=(["date_time"], np.zeros(time.size)),
                DLWR=(["date_time"], np.zeros(time.size)),
                APCP=(["date_time"], precipitation_flux*np.ones(time.size)),
                Temp=(["date_time"], 300*np.ones(time.size)),
                UGRD=(["date_time"], np.zeros(time.size)),
                VGRD=(["date_time"], np.zeros(time.size)),
                Press=(["date_time"], 1e5*np.ones(time.size)),
                SPFH=(["date_time"], np.zeros(time.size)),
            ),
            coords=dict(
                date_time=time,
            ),
            attrs=dict(
                description="Meteorological dataset from CHMI opendata.",
                time_step=time_step,
                time_interval=time[-1]-time[0]
            ),
        )

        # Setup CLM
        toy._run.Solver.LSM = "CLM"
        toy._run.Patch.top.BCPressure.Type = "OverlandFlow"
        toy._run.Patch.top.BCPressure.Cycle = "constant"
        toy._run.Patch.top.BCPressure.alltime.Value = 0.0
        toy._run.Solver.CLM.MetForcing = "1D"
        toy._run.Solver.CLM.MetFileName = "narr_1hr.txt"


        # Run and postprocess
        toy.run(init_pressure=pressure,
                met_data=ds,
                input_dir=TEST_DIR
                # precipitation_value=precipitation_flux,
                # start_time=0.0,
                # stop_time=stop_time,
                # time_step=model_time_step
        )

        assert_parflow_log(working_dir)
        assert_kalman_model_outputs(toy, current_time_step=stop_time)
        toy.save_pressure(output_path / "pressure.png")


if __name__ == "__main__":
    test_parflow_model_with_clm()
