import tempfile
from pathlib import Path
from hlavo.soil_parflow.parflow_model import ToyProblem

def test_parflow_model():

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Model initialization
        output_path = Path(tmp_dir)
        toy = ToyProblem(model_config, workdir=output_path / "output-toy")

        # Model parameters
        precipitation_flux = -0.0166
        stop_time = 200
        model_time_step = 0.025
        pressure = toy.make_linear_pressure(model_config)

        # Run and postprocess
        toy.run(init_pressure=pressure,
                precipitation_value=precipitation_flux,
                start_time=0,
                stop_time=stop_time,
                time_step=model_time_step)

        toy.save_pressure(output_path / "pressure.png")