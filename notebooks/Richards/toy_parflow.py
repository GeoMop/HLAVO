# Sample problem for Richards equation solved by PARFLOW
#(requires "pftools" Python package provided via pip)

from parflow import Run
from parflow.tools import settings
from parflow.tools.io import write_pfb, read_pfb
import numpy as np
import os, pathlib
from matplotlib import pyplot as plt

class ToyProblem:
    def __init__(self, workdir=None):
        # Define a toy problem for PARFLOW simulator
        self._run = Run("toy_richards", __file__)
        if workdir is not None:
            self._workdir = pathlib.Path(workdir)
            pathlib.Path.mkdir(self._workdir, exist_ok=True)
        else:
            self._workdir = pathlib.Path.cwd()


    def setup_config(self):
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
        self._run.ComputationalGrid.Lower.Z = -10.0

        self._run.ComputationalGrid.DX = 1.
        self._run.ComputationalGrid.DY = 1.
        self._run.ComputationalGrid.DZ = 0.5

        self._run.ComputationalGrid.NX = 1
        self._run.ComputationalGrid.NY = 1
        self._run.ComputationalGrid.NZ = 20

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
        self._run.Geom.domain.Lower.X = 0.0
        self._run.Geom.domain.Lower.Y = 0.0
        self._run.Geom.domain.Lower.Z = -10.0

        self._run.Geom.domain.Upper.X = 1.0
        self._run.Geom.domain.Upper.Y = 1.0
        self._run.Geom.domain.Upper.Z = 0.0

        self._run.Geom.domain.Patches = "left right front back bottom top"

        #-----------------------------------------------------------------------------
        # Permeability
        #-----------------------------------------------------------------------------
        self._run.Geom.Perm.Names = "domain"

        self._run.Geom.domain.Perm.Type = "Constant"
        self._run.Geom.domain.Perm.Value = 30.8 / 100 / 24 # 1.2833e-2 [cm/d] -> [m/h]

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
        self._run.TimingInfo.BaseUnit = 1.0e-4
        self._run.TimingInfo.StartCount = 0
        self._run.TimingInfo.StartTime = 0.0
        self._run.TimingInfo.StopTime = 48.0  # [h]
        self._run.TimingInfo.DumpInterval = -1
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
        self._run.Geom.domain.RelPerm.Alpha = 0.58
        self._run.Geom.domain.RelPerm.N = 2.4

        #---------------------------------------------------------
        # Saturation
        #---------------------------------------------------------
        self._run.Phase.Saturation.Type = "VanGenuchten"
        self._run.Phase.Saturation.GeomNames = "domain"
        self._run.Geom.domain.Saturation.Alpha = 0.58
        self._run.Geom.domain.Saturation.N = 3.7
        self._run.Geom.domain.Saturation.SRes = 0.06
        self._run.Geom.domain.Saturation.SSat = 0.47

        #-----------------------------------------------------------------------------
        # Boundary Conditions: Pressure
        #-----------------------------------------------------------------------------
        self._run.BCPressure.PatchNames = "bottom top"

        self._run.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
        self._run.Patch.bottom.BCPressure.Cycle = "constant"
        self._run.Patch.bottom.BCPressure.RefGeom = "domain"
        self._run.Patch.bottom.BCPressure.RefPatch = "bottom"
        self._run.Patch.bottom.BCPressure.alltime.Value = -2.0

        self._run.Patch.top.BCPressure.Type = "FluxConst"
        self._run.Patch.top.BCPressure.Cycle = "constant"
        self._run.Patch.top.BCPressure.alltime.Value = -2e-3

        #---------------------------------------------------------
        # Initial conditions: water pressure
        #---------------------------------------------------------
        self._run.ICPressure.Type = "HydroStaticPatch"
        self._run.ICPressure.GeomNames = "domain"
        self._run.Geom.domain.ICPressure.Value = -2.0
        self._run.Geom.domain.ICPressure.RefGeom = "domain"
        self._run.Geom.domain.ICPressure.RefPatch = "bottom"

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

        # === End Other required and unused parameters ===


    def set_init_pressure(self):
        # example of setting custom initial pressure

        # create vector of z-coordinates for data vector in ascending order
        nz = self._run.ComputationalGrid.NZ
        dz = self._run.ComputationalGrid.DZ
        z0 = self._run.ComputationalGrid.Lower.Z
        zz = np.linspace(z0,z0+(nz-1)*dz,nz)

        # define initial pressure data vector
        init_p = np.zeros((nz,1,1))
        init_p[:,0,0] = zz-2

        filename = "toy_richards.init_pressure.pfb"
        filepath = self._workdir / pathlib.Path(filename)
        write_pfb(str(filepath), init_p)

        self._run.ICPressure.Type = "PFBFile"
        self._run.Geom.domain.ICPressure.FileName = filename


    def set_porosity(self, z_values, porosity_values):
        # example of setting porosity by piecewise linear interpolation of given values

        # create vector of z-coordinates for data vector in ascending order
        nz = self._run.ComputationalGrid.NZ
        dz = self._run.ComputationalGrid.DZ
        z0 = self._run.ComputationalGrid.Lower.Z
        zz = np.linspace(z0,z0+(nz-1)*dz,nz)

        # interpolate porosity values
        por = np.zeros((nz,1,1))
        por[:,0,0] = np.interp(zz, z_values, porosity_values)

        filename = "toy_richards.porosity.pfb"
        filepath = self._workdir / pathlib.Path(filename)
        write_pfb(str(filepath), por)

        self._run.Geom.domain.Porosity.Type = "PFBFile"
        self._run.Geom.domain.Porosity.FileName = filename


    def run(self):
        self._run.write(file_format='yaml')
        self._run.run(working_directory=self._workdir)


    def load_yaml(self, yaml_file):
        ## Create a Run object from a .yaml file
        self._run = Run.from_definition(yaml_file)


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
        plt.imshow(np.flip(pressure), aspect='auto')
        nticks = int(ntimes/10)
        plt.yticks( np.arange(ntimes)[::nticks], np.flip(data.times[::nticks]) )
        nzticks = int(nz/10)
        plt.xticks( np.arange(nz)[1::nzticks], np.cumsum(data.dz)[1::nzticks] )
        plt.colorbar()
        plt.title("pressure")
        plt.xlabel("depth [m]")
        plt.ylabel("time [h]")
        plt.savefig(image_file)

        settings.set_working_directory(cwd)


    def save_porosity(self, image_file):
        cwd = settings.get_working_directory()
        settings.set_working_directory(self._workdir)

        # Get the DataAccessor object corresponding to the Run object
        data = self._run.data_accessor
        data.time = 0

        ntimes = len(data.times)
        nz = data.computed_porosity.shape[0]
        porosity = np.zeros((ntimes, nz))

        # Iterate through the timesteps of the DataAccessor object
        # i goes from 0 to n_timesteps - 1
        for i in data.times:
            porosity[data.time,:] = data.computed_porosity.reshape(nz)
            data.time += 1

        plt.clf()
        plt.imshow(np.flip(porosity), aspect='auto')
        nticks = int(ntimes/10)
        plt.yticks( np.arange(ntimes)[::nticks], np.flip(data.times[::nticks]) )
        nzticks = int(nz/10)
        plt.xticks( np.arange(nz)[1::nzticks], np.cumsum(data.dz)[1::nzticks] )
        plt.colorbar()
        plt.title("porosity")
        plt.xlabel("depth [m]")
        plt.ylabel("time [h]")
        plt.savefig(image_file)

        settings.set_working_directory(cwd)


toy = ToyProblem(workdir="output-toy")
toy.setup_config()
toy.set_init_pressure()
toy.set_porosity([-10,-5,0], [0.1, 1, 0.5])
toy.run()
toy.save_pressure("pressure.png")
toy.save_porosity("porosity.png")
