# Sample problem for Richards equation solved by PARFLOW
#(requires "pftools" Python package provided via pip)

from parflow import Run
from parflow.tools import settings
from parflow.tools.io import write_pfb
import numpy as np
import os, pathlib
from parflow.tools.fs import get_absolute_path
import pytest


class ToyProblem:
    def __init__(self, workdir=None):
        # Define a toy problem for PARFLOW simulator
        self._run = Run("toy_richards", __file__)
        if workdir is not None:
            self._workdir = pathlib.Path(workdir)
            pathlib.Path.mkdir(self._workdir, exist_ok=True, parents=True)
        else:
            self._workdir = pathlib.Path.cwd()

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

    def make_linear_pressure(self, init_pressure):
        p_top, p_bot = init_pressure
        nz = self._run.ComputationalGrid.NZ
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
        self._run.ComputationalGrid.Lower.X = 0.0
        self._run.ComputationalGrid.Lower.Y = 0.0

        self._run.ComputationalGrid.DX = 1
        self._run.ComputationalGrid.DY = 1

        self._run.ComputationalGrid.NX = 1
        self._run.ComputationalGrid.NY = 1

        self._run.ComputationalGrid.Lower.Z = -135
        self._run.ComputationalGrid.DZ = 9
        self._run.ComputationalGrid.NZ = 15
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

        self._run.Geom.domain.Lower.X = 0.0
        self._run.Geom.domain.Upper.X = 1.0

        self._run.Geom.domain.Lower.Y = 0.0
        self._run.Geom.domain.Upper.Y = 1.0

        self._run.Geom.domain.Lower.Z = low_z
        self._run.Geom.domain.Upper.Z = up_z

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
        self._run.TimingInfo.DumpInterval = 1

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
        return data_accessor._pfb_to_array(file_name)

    def get_times(self):
        return self._run.data_accessor.times



# ============================================================================
# Pytest fixtures
# ============================================================================

@pytest.fixture
def toy(tmp_path):
    toy_problem = ToyProblem(workdir="output-toy")
    toy_problem.setup_config()
    init_pressure = toy_problem.make_linear_pressure([-40, -100])
    toy_problem.set_init_pressure(init_pressure)
    toy_problem.run(init_pressure, precipitation_value=-0.09523)
    return toy_problem


# ============================================================================
# Unit tests
# ============================================================================
def test_get_nodes_z(toy):
    nodes = toy.get_nodes_z()
    assert len(nodes) == 16
    assert nodes[0] == -135.0
    assert nodes[-1] == 0


def test_get_element_centers_z(toy):
    centers = toy.get_el_centers_z()
    nz = toy._run.ComputationalGrid.NZ
    assert len(centers) == nz


def test_make_linear_pressure(toy):
    p = toy.make_linear_pressure([-40, -100])
    assert len(p) == 15
    assert p[0] == -100.
    assert p[-1] == -40.
    assert np.all(np.diff(p) > 0)


def test_get_data(toy):
    nz = toy._run.ComputationalGrid.NZ
    data = toy.get_data(current_time_step=0, data_name="pressure")
    assert data.shape == (nz,)

    data = toy.get_data(current_time_step=0, data_name="moisture")
    assert data.shape == (nz,)

    data = toy.get_data(current_time_step=0, data_name="velocity")
    assert data.shape == (nz + 1,)


# toy = ToyProblem(workdir="output-toy")
# toy.setup_config()
# init_pressure = toy.make_linear_pressure([-40, -100])
# toy.set_init_pressure(init_pressure)
# #toy.set_porosity([-10,-5,0], [0.1, 1, 0.5])
# toy.run(init_pressure, precipitation_value=-0.09523)
# new_pressure = toy.get_data(current_time_step=20, data_name="pressure")
# print("pressure ", new_pressure)
