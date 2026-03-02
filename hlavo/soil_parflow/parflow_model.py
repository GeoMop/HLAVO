# Sample problem for Richards equation solved by PARFLOW
#(requires "pftools" Python package provided via pip)

from parflow import Run
from parflow.tools import settings
from parflow.tools.io import write_pfb
#from soil_model import evapotranspiration_fce
import numpy as np
import os, pathlib
from matplotlib import pyplot as plt
from hlavo.soil_parflow.abstract_model import AbstractModel
from hlavo.misc.auxiliary_functions import set_nested_attr, set_nested_attrs
from parflow.tools.fs import get_absolute_path


class ToyProblem(AbstractModel):
    def __init__(self, config, workdir=None):
        # Define a toy problem for PARFLOW simulator
        self._run = Run("toy_richards", __file__)
        if workdir is not None:
            self._workdir = pathlib.Path(workdir)
            pathlib.Path.mkdir(self._workdir, exist_ok=True, parents=True)
        else:
            self._workdir = pathlib.Path.cwd()

        self.key_to_parflow_param = config["params"]
        self.setup_config(config["static_params"])
        # Check PARFLOW installation
        parflow_dir = os.environ.get('PARFLOW_DIR', None)
        assert parflow_dir is not None, "The PARFLOW_DIR environment variable is not set."
        parflow_path = pathlib.Path(parflow_dir)
        # Check if the directory exists
        assert parflow_path.is_dir(), f"The PARFLOW_DIR environment variable is set, but the directory does not exist: {parflow_path}"

    def get_nodes_z(self):
        """
        Grid nodes
        Bottom are first.
        :return:
        """
        nz = self._run.ComputationalGrid.NZ
        dz = self._run.ComputationalGrid.DZ
        return (self._run.ComputationalGrid.Lower.Z + np.linspace(0.0, nz * dz, nz+1))

    def get_el_centers_z(self):
        """
        Center points of finite volumes.
        Bottom are first.
        :return:
        """
        nodes_z = self.get_nodes_z()
        return (nodes_z[1:] + nodes_z[:-1]) / 2.0

    def make_linear_pressure(self, cfg):
        p_top, p_bot = cfg['init_pressure']
        nz = self._run.ComputationalGrid.NZ
        # dz = self._run.ComputationalGrid.DZ
        #zz = np.linspace(p_top, p_bot, nz)
        zz = np.linspace(p_bot, p_top, nz)
        return zz


    def setup_config(self, static_params_dict={}):
        #-----------------------------------------------------------------------------
        # File input version number
        #-----------------------------------------------------------------------------
        self._run.FileVersion = 4

        #-----------------------------------------------------------------------------
        # Process Topology
        #-----------------------------------------------------------------------------
        self._run.Process.Topology.P = 1
        self._run.Process.Topology.Q = 1
        self._run.Process.Topology.R = 1

        #-----------------------------------------------------------------------------
        # Computational Grid
        #-----------------------------------------------------------------------------

        # Set the physical size of the domain in meters
        obj = self._run.ComputationalGrid
        fixed = {
            "Lower.X":0.0, "DX": 1, "NX":1,
            "Lower.Y": 0.0, "DY": 1, "NY": 1}
        set_nested_attrs(obj, fixed)
        z_grid_dict = {("ComputationalGrid." + k):static_params_dict["ComputationalGrid"][k]
                       for k in ["Lower.Z", "DZ", "NZ"]}
        set_nested_attrs(self._run, z_grid_dict)
        #-----------------------------------------------------------------------------
        # The Names of the GeomInputs
        #-----------------------------------------------------------------------------
        self._run.GeomInput.Names = "domain_input"
        #-----------------------------------------------------------------------------
        # Domain Geometry Input
        #-----------------------------------------------------------------------------
        self._run.GeomInput.domain_input.InputType = "Box"
        self._run.GeomInput.domain_input.GeomName = "domain"

        #-----------------------------------------------------------------------------
        # Domain Geometry
        #-----------------------------------------------------------------------------
        low_z = self._run.ComputationalGrid.Lower.Z
        up_z = low_z  + self._run.ComputationalGrid.DZ * self._run.ComputationalGrid.NZ
        domain_dict = {
            "Lower.X": 0.0, "Upper.X": 1.0,
            "Lower.Y": 0.0, "Upper.Y": 1.0,
            "Lower.Z": low_z, "Upper.Z": up_z,
        }
        set_nested_attrs(self._run.Geom.domain, domain_dict)
        self._run.Geom.domain.Patches = "left right front back bottom top"

        #-----------------------------------------------------------------------------
        # Permeability
        #-----------------------------------------------------------------------------
        if "Perm" in static_params_dict:
            self.set_perm(list(static_params_dict["Perm"].keys()), list(static_params_dict["Perm"].values()))
        else:
            self._run.Geom.Perm.Names = "domain"
            self._run.Geom.domain.Perm.Type = "Constant"
            self._run.Geom.domain.Perm.Value = 0.0737 #30.8 / 100 / 24 #30.8 / 100 / 24 # 1.2833e-2 [cm/d] -> [m/h]
            self._run.Perm.TensorType = "TensorByGeom"
            self._run.Geom.Perm.TensorByGeom.Names = "domain"
            self._run.Geom.domain.Perm.TensorValX = 1.0
            self._run.Geom.domain.Perm.TensorValY = 1.0
            self._run.Geom.domain.Perm.TensorValZ = 1.0

        #-----------------------------------------------------------------------------
        # Specific Storage
        #-----------------------------------------------------------------------------
        # specific storage does not figure into the impes (fully sat) case but we still
        # need a key for it
        self._run.SpecificStorage.Type = "Constant"
        self._run.SpecificStorage.GeomNames = ""
        self._run.Geom.domain.SpecificStorage.Value = 1.0

        #-----------------------------------------------------------------------------
        # Phases
        #-----------------------------------------------------------------------------
        self._run.Phase.Names = "water"

        self._run.Phase.water.Density.Type = "Constant"
        self._run.Phase.water.Density.Value = 1.0

        self._run.Phase.water.Viscosity.Type = "Constant"
        self._run.Phase.water.Viscosity.Value = 1.0

        self._run.Phase.water.Mobility.Type = "Constant"
        self._run.Phase.water.Mobility.Value = 1.0

        #-----------------------------------------------------------------------------
        # Gravity
        #-----------------------------------------------------------------------------
        self._run.Gravity = 1.0

        #-----------------------------------------------------------------------------
        # Setup timing info
        #-----------------------------------------------------------------------------
        self._run.TimingInfo.BaseUnit = 1 #1.0e-4
        self._run.TimingInfo.StartCount = 0
        self._run.TimingInfo.StartTime = 0.0
        self._run.TimingInfo.StopTime = 60 #48.0  # [h]
        self._run.TimingInfo.DumpInterval = 1
        self._run.TimeStep.Type = "Constant"
        self._run.TimeStep.Value = 2.5e-2     # [h]

        #-----------------------------------------------------------------------------
        # Time Cycles
        #-----------------------------------------------------------------------------
        self._run.Cycle.Names = "constant"
        self._run.Cycle.constant.Names = "alltime"
        self._run.Cycle.constant.alltime.Length = 1
        self._run.Cycle.constant.Repeat = -1

        #-----------------------------------------------------------------------------
        # Porosity
        #-----------------------------------------------------------------------------
        self._run.Geom.Porosity.GeomNames = "domain"
        self._run.Geom.domain.Porosity.Type = "Constant"
        self._run.Geom.domain.Porosity.Value = 1.0

        #-----------------------------------------------------------------------------
        # Domain
        #-----------------------------------------------------------------------------
        self._run.Domain.GeomName = "domain"

        #-----------------------------------------------------------------------------
        # Relative Permeability
        #-----------------------------------------------------------------------------
        self._run.Phase.RelPerm.Type = "VanGenuchten"
        self._run.Phase.RelPerm.GeomNames = "domain"
        self._run.Geom.domain.RelPerm.Alpha = 0.75 #0.58
        self._run.Geom.domain.RelPerm.N = 1.89 #3.7

        #---------------------------------------------------------
        # Saturation
        #---------------------------------------------------------
        self._run.Phase.Saturation.Type = "VanGenuchten"
        self._run.Phase.Saturation.GeomNames = "domain"
        #print("self._run.Geom.domain.RelPerm.Alpha ", self._run.Geom.domain.RelPerm.Alpha)
        self._run.Geom.domain.Saturation.Alpha = 0.75 #0.58  #self._run.Geom.domain.RelPerm.Alpha  # 0.58
        self._run.Geom.domain.Saturation.N = 1.89 #3.7  #self._run.Geom.domain.RelPerm.N  # 3.7
        self._run.Geom.domain.Saturation.SRes = 0.065 #0.06
        self._run.Geom.domain.Saturation.SSat = 0.41 #0.47

        #-----------------------------------------------------------------------------
        # Boundary Conditions: Pressure
        #-----------------------------------------------------------------------------
        self._run.BCPressure.PatchNames = "bottom top"
        self._run.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
        self._run.Patch.bottom.BCPressure.Cycle = "constant"
        self._run.Patch.bottom.BCPressure.RefGeom = "domain"
        self._run.Patch.bottom.BCPressure.RefPatch = "bottom"
        self._run.Patch.bottom.BCPressure.alltime.Value = static_params_dict["BCPressure_bottom"] if "BCPressure_bottom" in static_params_dict else -50

        # self._run.Patch.bottom.BCPressure.Type = "FluxConst"
        # self._run.Patch.bottom.BCPressure.Cycle = "constant"
        # self._run.Patch.bottom.BCPressure.alltime.Value = -5

        self._run.Patch.top.BCPressure.Type = "FluxConst"
        self._run.Patch.top.BCPressure.Cycle = "constant"
        self._run.Patch.top.BCPressure.alltime.Value = 0 #-2e-2 #-1.3889 * 10**-6  # 5 mm/h #-2e-2 #-2e-2 #-2e-3 # set in [m/s]

        #---------------------------------------------------------
        # Initial conditions: water pressure
        #---------------------------------------------------------
        # self._run.ICPressure.Type = "HydroStaticPatch"
        self._run.ICPressure.GeomNames = "domain"
        # self._run.Geom.domain.ICPressure.Value = -2.0
        # self._run.Geom.domain.ICPressure.RefGeom = "domain"
        # self._run.Geom.domain.ICPressure.RefPatch = "bottom"

        #-----------------------------------------------------------------------------
        # Phase sources:
        #-----------------------------------------------------------------------------
        self._run.PhaseSources.water.Type = "Constant"
        self._run.PhaseSources.water.GeomNames = "domain"
        self._run.PhaseSources.water.Geom.domain.Value = 0.0

        #-----------------------------------------------------------------------------
        # Set solver parameters
        #-----------------------------------------------------------------------------
        self._run.Solver = "Richards"
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

        self._run.Solver.PrintVelocities = True

        #self._run.Solver.Pressure.FileName = "pressure.out"
        #self._run.Solver.Saturation.FileName = "saturation.out"
        # === Other required, but unused parameters ===

        #---------------------------------------------------------
        # Topo slopes in x-direction
        #---------------------------------------------------------
        # topo slopes do not figure into the impes (fully sat) case but we still
        # need keys for them
        self._run.TopoSlopesX.Type = "Constant"
        self._run.TopoSlopesX.GeomNames = ""
        self._run.TopoSlopesX.Geom.domain.Value = 0.0

        #---------------------------------------------------------
        # Topo slopes in y-direction
        #---------------------------------------------------------
        self._run.TopoSlopesY.Type = "Constant"
        self._run.TopoSlopesY.GeomNames = ""
        self._run.TopoSlopesY.Geom.domain.Value = 0.0

        #---------------------------------------------------------
        # Mannings coefficient
        #---------------------------------------------------------
        # mannings roughnesses do not figure into the impes (fully sat) case but we still
        # need a key for them
        self._run.Mannings.Type = "Constant"
        self._run.Mannings.GeomNames = ""
        self._run.Mannings.Geom.domain.Value = 0.

        #-----------------------------------------------------------------------------
        # Wells
        #-----------------------------------------------------------------------------
        self._run.Wells.Names = ""

        #-----------------------------------------------------------------------------
        # Contaminants
        #-----------------------------------------------------------------------------
        self._run.Contaminants.Names = ""

        #-----------------------------------------------------------------------------
        # Exact solution specification for error calculations
        #-----------------------------------------------------------------------------
        self._run.KnownSolution = "NoKnownSolution"

    def set_perm(self, z_coords, perms):
        # set piecewise constant permeability (z_coords assumed to be in descending order)
        regions_str = " ".join(f"region{i}" for i, z in enumerate(z_coords))
        self._run.GeomInput.Names = "domain_input " + regions_str
        print(self._run.GeomInput.Names)

        z_coords.append(self._run.Geom.domain.Lower.Z)
        for i, z in enumerate(perms):
            reg_name = f"region{i}"
            self._run.GeomInput[reg_name].InputType = 'Box'
            self._run.GeomInput[reg_name].GeomName = reg_name
            g = self._run.Geom[reg_name]
            g.Upper.X, g.Upper.Y, g.Upper.Z = 1.0, 1.0, z_coords[i]
            g.Lower.X, g.Lower.Y, g.Lower.Z = 0.0, 0.0, z_coords[i + 1]

            # Sufficient for 'kolona' experiments
            #@TODO: make it more general
            if i == 0:
                self.key_to_parflow_param["vG_K_s"] = "Geom.{}.Perm.Value".format(reg_name)

        self._run.Geom.Perm.Names = regions_str
        self._run.Perm.TensorType = 'TensorByGeom'
        for i, p in enumerate(perms):
            g = self._run.Geom[f"region{i}"]
            g.Perm.Type = 'Constant'
            g.Perm.Value = p

        self._run.Perm.TensorType = "TensorByGeom"
        self._run.Geom.Perm.TensorByGeom.Names = "domain"
        self._run.Geom.domain.Perm.TensorValX = 1.0
        self._run.Geom.domain.Perm.TensorValY = 1.0
        self._run.Geom.domain.Perm.TensorValZ = 1.0

    def set_init_pressure(self, init_p, working_dir=None):
        # setting custom initial pressure

        if working_dir is None:
            working_dir = self._workdir

        filename = "toy_richards.init_pressure.pfb"
        filepath = working_dir / pathlib.Path(filename)
        init_p = init_p[:, None, None]
        write_pfb(str(filepath), init_p)

        self._run.ICPressure.Type = "PFBFile"
        self._run.Geom.domain.ICPressure.FileName = filename

    def set_porosity(self, z_values, porosity_values):
        # example of setting porosity by piecewise linear interpolation of given values
        nz = self._run.ComputationalGrid.NZ
        dz = self._run.ComputationalGrid.DZ
        zz = np.linspace(0, -(nz - 1) * dz, nz)

        por = np.zeros((nz, 1, 1))
        por[:, 0, 0] = np.interp(zz, z_values, porosity_values)

        filename = "toy_richards.porosity.pfb"
        filepath = self._workdir / pathlib.Path(filename)
        write_pfb(str(filepath), por)

        self._run.Geom.domain.Porosity.Type = "PFBFile"
        self._run.Geom.domain.Porosity.FileName = filename

    def run(self, init_pressure, precipitation_value, state_params=None, start_time=0, stop_time=20, time_step=0.025, working_dir=None):
        import shutil
        import os
        if working_dir is None:
            working_dir = self._workdir
        shutil.rmtree(working_dir)
        os.makedirs(working_dir)

        if state_params is not None:
            self.set_dynamic_params(state_params)
        self._run.Patch.top.BCPressure.alltime.Value = precipitation_value
        self.set_init_pressure(init_pressure, working_dir=working_dir)
        self._run.TimingInfo.StartTime = start_time
        self._run.TimingInfo.StopTime = stop_time
        self._run.TimeStep.Value = time_step
        self._run.TimingInfo.DumpInterval = -1

        print("start: {} stop: {} step: {}".format(start_time, stop_time, time_step))

        self._run.run(working_directory=str(working_dir))
        self._run.write(file_format='yaml')

        settings.set_working_directory(working_dir)

    def get_data(self, current_time_step, data_name="pressure"):
        data = self._run.data_accessor
        data.time = current_time_step

        if data_name == "pressure":
            return data.pressure[:, 0, 0]
        elif data_name == "moisture":
            return data.saturation[:, 0, 0]
        elif data_name == "velocity":
            return self.get_velocity(data_accessor=data)[:, 0, 0]
        else:
            raise NotImplemented("This method returns 'pressure', 'saturation' or 'velocity' only")

    def get_velocity(self, data_accessor):
        file_name = get_absolute_path(f'{data_accessor._name}.out.velz.{data_accessor._ts}.pfb')
        velocity = data_accessor._pfb_to_array(file_name)
        return data_accessor._pfb_to_array(file_name)

    def get_times(self):
        return self._run.data_accessor.times

    # def get_space_step(self):
    #     return self._run.ComputationalGrid.DZ

    def load_yaml(self, yaml_file):
        ## Create a Run object from a .yaml file
        self._run = Run.from_definition(yaml_file)

    # def save_porosity(self, image_file):
    #     cwd = settings.get_working_directory()
    #     settings.set_working_directory(self._workdir)
    #
    #     # Get the DataAccessor object corresponding to the Run object
    #     data = self._run.data_accessor
    #     data.time = 0
    #
    #     ntimes = len(data.times)
    #     nz = data.computed_porosity.shape[0]
    #     porosity = np.zeros((ntimes, nz))
    #
    #     # Iterate through the timesteps of the DataAccessor object
    #     # i goes from 0 to n_timesteps - 1
    #     for i in data.times:
    #         porosity[data.time, :] = data.computed_porosity.reshape(nz)
    #         data.time += 1
    #
    #     plt.clf()
    #     plt.imshow(np.flip(porosity), aspect='auto')
    #     nticks = int(ntimes / 10)
    #     plt.yticks(np.arange(ntimes)[::nticks], np.flip(data.times[::nticks]))
    #     nzticks = int(nz / 10)
    #     plt.xticks(np.arange(nz)[1::nzticks], np.cumsum(data.dz)[1::nzticks])
    #     plt.colorbar()
    #     plt.title("porosity")
    #     plt.xlabel("depth [m]")
    #     plt.ylabel("time [h]")
    #     plt.savefig(image_file)
    #
    #     settings.set_working_directory(cwd)

    def set_dynamic_params(self, model_params):
        for key, val in model_params.items():
            if not key in self.key_to_parflow_param:
                continue
            targets = self.key_to_parflow_param[key]
            if isinstance(targets, str):
                targets = [targets]
            for target in targets:
                set_nested_attr(self._run, target, model_params[key])

            #print("params: {}, mean: {}, std: {}".format(params, mean, std))

            # if params == "Patch.top.BCPressure.alltime.Value":
            #     print("precipitation list time step: {} value: {}".format(time_step,
            #           model_config["precipitation_list"][time_step]))
            #     if model_config["precipitation_list"][time_step] == 0:
            #         value = 0
            #     # elif model_config["precipitation_list"][time_step-1] == 0:
            #     #     print("model params std ", model_config["params"]["std"])
            #     #     value = add_noise([model_config["precipitation_list"][time_step]],
            #     #               distr_type=kalman_config["noise_distr_type"],
            #     #               std=model_config["params"]["std"][idx])
            #     value += et_per_time

            # if params == "Geom.domain.Saturation.Alpha":
            #     set_nested_attr(self._run, "Geom.domain.RelPerm.Alpha", mean)
            #
            # if params == "Geom.domain.Saturation.N":
            #     set_nested_attr(self._run, "Geom.domain.RelPerm.N", mean)

            #model_params_new_values.append(mean)

        #return model_params_new_values

    def save_pressure(self, image_file):
        cwd = settings.get_working_directory()
        settings.set_working_directory(self._workdir)

        # Get the DataAccessor object corresponding to the Run object
        data = self._run.data_accessor
        data.time = 0

        ntimes = len(data.times)
        nz = data.pressure.shape[0]
        pressure = np.zeros((ntimes, nz))

        # Iterate through the timesteps of the DataAccessor object
        # i goes from 0 to n_timesteps - 1
        for i in data.times:
            pressure[data.time,:] = data.pressure.reshape(nz)
            data.time += 1


        plt.clf()
        plt.imshow(np.flip(pressure), aspect='auto', extent=[0, np.sum(data.dz), 0, data.times[-1]])

        plt.colorbar()
        plt.title("pressure")
        plt.xlabel("depth [m]")
        plt.ylabel("time [h]")
        plt.savefig(image_file)
        plt.show()

        settings.set_working_directory(cwd)

    def plot_pressure(self, pressure, times):
        ntimes = pressure.shape[0]
        # self.save_pressure("pressure.png")
        # return

        el_centers = self.get_el_centers_z()
        # print("el centers z ", el_centers)

        print("pressure ", pressure.shape)
        print("times ", times)
        # print("pressure ", np.max(pressure))

        #times = list(np.cumsum(times))
        print("cumul times ", times)

        #dz = self._run.ComputationalGrid.DZ

        plt.imshow(pressure, aspect='auto', cmap='viridis', origin='lower')
        # Create the plot
        #plt.imshow(pressure, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Pressure')
        plt.xlabel('Depth [cm]')
        plt.ylabel('Time [min]')
        plt.title('Pressure Distribution')

        # Set x-ticks
        num_ticks = 5  # Adjust as needed
        indices = np.linspace(0, len(el_centers) - 1, num_ticks, dtype=int)  # Select evenly spaced indices
        plt.xticks(indices, np.round(el_centers[indices], 2))

        num_ticks = 6  # Number of ticks
        num_ticks = min(num_ticks, len(times))  # Prevent out-of-bounds
        indices = np.linspace(0, len(times) - 1, num_ticks, dtype=int)  # Generate indices
        times = np.array(times)  # Ensure times is a NumPy array
        plt.yticks(indices, times[indices])  # Apply cumulative sum as labels

        # Show the plot
        plt.savefig("pressure.pdf")
        plt.show()

        np.savez_compressed("plot_el_centers_data", data=el_centers)
        np.savez_compressed("plot_pressure_data", data=pressure)
        #nticks = int(ntimes / 10)
        # print("n ticks ", nticks)

        # from matplotlib import ticker
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((-1, 1))

        #nz = pressure.shape[1]

        # plt.yticks(np.arange(ntimes)[::nticks], np.flip(data.times[::nticks]))
        #nzticks = int(nz / 10)
        # print("np.arange(nz)[1::nzticks] ", np.arange(nz)[1::nzticks])
        # print("np.cumsum(data.dz) ", np.cumsum(data.dz))
        # print("np.cumsum(data.dz)[1::nzticks] ", np.cumsum(data.dz)[1::nzticks])
        # print("nzticks ", nzticks)
        # plt.xticks(np.arange(nz)[1::nzticks], np.cumsum(data.dz)[1::nzticks] )
        #plt.xticks(list(np.arange(nz)[1::nzticks]), list(np.cumsum(data.dz)[1::nzticks]))
        #plt.xticks(np.arange(nz)[1::nzticks], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2])
        # plt.colorbar()
        # plt.title("pressure")
        # plt.xlabel("depth [m]")
        # plt.ylabel("time [h]")
        # plt.savefig("ref_pressure.pdf")
        # plt.show()

    def plot_saturation(self, saturation, times):
        ntimes = saturation.shape[0]
        # self.save_pressure("pressure.png")
        # return

        el_centers = self.get_el_centers_z()
        # print("el centers z ", el_centers)
        #
        # print("pressure ", saturation)
        # print("pressure ", np.max(saturation))

        #dz = self._run.ComputationalGrid.DZ

        #times = list(np.cumsum(times))

        plt.imshow(saturation, aspect='auto', cmap='viridis', origin='lower')
        # Create the plot
        #plt.imshow(pressure, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Saturation')
        plt.xlabel('Depth [cm]')
        plt.ylabel('Time [min]')
        plt.title('Saturation Distribution')

        # Set x-ticks
        num_ticks = 5  # Adjust as needed
        indices = np.linspace(0, len(el_centers) - 1, num_ticks, dtype=int)  # Select evenly spaced indices
        plt.xticks(indices, np.round(el_centers[indices], 2))

        num_ticks = 6  # Number of ticks
        num_ticks = min(num_ticks, len(times))  # Prevent out-of-bounds
        indices = np.linspace(0, len(times) - 1, num_ticks, dtype=int)  # Generate indices
        times = np.array(times)  # Ensure times is a NumPy array
        plt.yticks(indices, times[indices])  # Apply cumulative sum as labels

        # Show the plot
        plt.savefig("saturation.pdf")
        plt.show()

        np.savez_compressed("plot_saturation_data", data=saturation)
        #nticks = int(ntimes / 10)
        # print("n ticks ", nticks)

        # from matplotlib import ticker
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((-1, 1))

        #nz = pressure.shape[1]

        # plt.yticks(np.arange(ntimes)[::nticks], np.flip(data.times[::nticks]))
        #nzticks = int(nz / 10)
        # print("np.arange(nz)[1::nzticks] ", np.arange(nz)[1::nzticks])
        # print("np.cumsum(data.dz) ", np.cumsum(data.dz))
        # print("np.cumsum(data.dz)[1::nzticks] ", np.cumsum(data.dz)[1::nzticks])
        # print("nzticks ", nzticks)
        # plt.xticks(np.arange(nz)[1::nzticks], np.cumsum(data.dz)[1::nzticks] )
        #plt.xticks(list(np.arange(nz)[1::nzticks]), list(np.cumsum(data.dz)[1::nzticks]))
        #plt.xticks(np.arange(nz)[1::nzticks], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2])
        # plt.colorbar()
        # plt.title("pressure")
        # plt.xlabel("depth [m]")
        # plt.ylabel("time [h]")
        # plt.savefig("ref_pressure.pdf")
        # plt.show()


# toy = ToyProblem(workdir="output-toy")
# toy.setup_config()
# toy.set_init_pressure()
# toy.set_porosity([-10,-5,0], [0.1, 1, 0.5])
# toy.run()
# toy.save_pressure("pressure.png")
