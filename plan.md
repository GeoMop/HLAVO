# Plan to complete composed and 3D model

## Rules

Slightly modified rules apply for this sprint. 
You are allowed and encouraged to create a single commit with its own branch for each fo the main steps.
You can make separate commits for substantial parts of the work on your own. 

The branches follows a single commit chain without actual branching. Idea is that I will then follow that chain 
later on and merge iteratively smaller bits that I can check.

You can only ask additional questions immediately after starting this plan, never ask once you start working on it.
So overview the plan and test your environment that modeflowapi works for you. Try a minimalistic usage so that it actually call the dynamic library
and verify it runs. I already checked that the package is indeed available.

## Answers to the questions
  1. AGENTS.md says never stage or commit, but plan.md says I’m allowed and encouraged to make branch/commit checkpoints. Which wins for this work?
  
     Follow plan for this goal. I ask you to stage and commit but as I instructed put there the branch for each completed milestone as JB_step_01, ...
     
  2. Should I treat the current dirty worktree as the intended baseline and fix forward from it, including current tracked/untracked composed files, or s
     hould I first reconcile against the older handoff state in STATUS.md?
     
     Good point. Run the update-status skill, but STATUS is not meant to provide authoritative status of the code base, its rather 
     track of history and should in particular track important bug fixes and design choices.
     So, run the skill but check what runs yourself since this branch is a bit broken after a complex merge with kalman developlment.
     I onl managed to make tests/model_1d to work. 
    
     
  3. Do you want me to execute only milestones 1-3 first, then stop for review, or continue into milestones 4-7 after milestone 3 passes?
 
     Just continue to 4 and down to 7, that is just a note about the state that starts to be relevant for the steps 4 - 7.
  
  4. For composed config, should the canonical 1D site input be model_1d.site_ids: [1, ...], or model_1d.sites with longitude/latitude entries? The
     current code and test disagree.
     
     The first option, model_1d.site_ids [ints] with longitude latitude given by 1D  reading parflow node of chmi_stations.
     This is already applied in tests/model_1d
     
  5. For mocks, should KalmanMock / KalmanScalingMock live beside the real Kalman 1D model, or inside hlavo/composed as coupling-test support classes?
  
     Pleas put 1D models into kalman even if those are not kalman in fact but rename them to the pattern SurfaceMocl SurfaceScalingMock and SurfaceKalman.
  
  6. For the simple file writer in milestone 2, what format do you prefer: JSONL, CSV, or a tiny xarray/netCDF-style artifact? For tests, JSONL is the
     simplest to count entries.
     
     Go on with JSONL for readability.
     
  7. For ZARR writer tests, should I generate a minimal local wells store fixture from wells_schema.yaml, or reuse an existing test store if available?
     
     Just read the form the existing wells store given by the STORE_URL in the schema attrs. Use zarr-fuse.open_store.
     
  8. In milestone 3, is flopy acceptable for building the simple cube MODFLOW simulation inside the test setup? It is installed and worked in the smoke
     test.

     Its fine to use flopy as is it fine for building the simulation. I only want to avoid repeated execution of the simulation as coupled with the surface model and rather use the modflowapi and out own loop.
     
## Milestones

1. Implement super simple unit test for the composed/model_composed.py that just couple model 1D mock (KalmanMock) and model 3D mock, 
   configure them through the config.yaml , follow tests/model_1d. 
   Modify existing mocks as needed, use some simple calculation so that you can check the result of it.
   Preserve datatime64 time tracking, but set the timestep so that you only perform few iterations to check that message passing and 
   synchronization.
   
   Milestone: Test works and effectively test already existing functionality in model_composed, worker_1d and model_3d
   You should not need to modify exiting model_* code, but of course apply fixes if you find bugs. You can also add a minimalistic inspection
   on the way, but remove them before end.
   
2. Goal test reading ZARR inputs and implement ZARR outputs for the coupled model.
   
   Implement KalmanScalingMock: it gets meteo and profiles from Model1D (see the stepmethod). If profiles are available use  mean moisture to set the scaling factor: 
   form 0.1 (zero moisture or not available) up to 0.8 (saturated profiles); return recharge given as scaling_factor * mean percipitation (in mm / m / day) over 48h window
   
   Implement Model3DDelay backend mock that 
   - initial accumulator (water_level) = -60
   - add all recharges * time_step to its inner accumulator
   - simulate drainage: acumulator = - positive_part_of(acumulator - -60) * time_step * 0.1
   - return water_level
   
   Implement the data writer class in two variants one simply writing to a file, other writing to the zarr-fuse store 
   designed by the hlavo/schemas/simulation_schema.yaml.
   The writer will be created in Model3D and it will collect data passed through the queues. These data will go into the site_prediction node,
   writing to the pressure_head (comming from the 3d model) and velocity (m/s , Darcy velocity comming from the 1D model). Fill coordinate "calibration" by the 
   initial datatime  when the writer was created so each simulation will currently has its own "calibration" item.
   The writer would read the wells dataset (using wells_schema.yaml) first to get positions and open intervals of the the wells, these positions will be later used by Model3D to ask the 3D backend 
   for reading the pressure_head values on the wells. Model3D will pass that to the writer and it will write to the well_prediction node of the simulation dataset.
   You may possibly need some tweaks on the simulation schema, this will actually be its first usage.
   
   Create a unit test to test a simulation of a single month from 1.3. 2025, that interval should have data from both the meteo and profiles.
   Use config.yaml in common format to instantiate the new mock classes, pass their parameters from there.
   
   Milestone: All implementations done, two unit test variants: for the file writerand for the ZARR writer, but overwrite URI to store into a local ZARR store.
   For the file writer do through test that correct number of entries of particular variables is there, do not check actual values. For the local zARR store jsut check sizes of coords.
   zarr-fuse open_storeshould allow overriding the schema URI by the system variable ZF_STORE_URL.If it will not work make the scehma copy and do the chenge in it.
   
   
3. make a basic unit test to test modflowapi fitting into structure of the composed.model_composed
   - you use the worker 1D and 1d model mock
   - you create a new Model3DAPI backend implementation having:
      a) initialization method called shortly after construction for building modflow model (just simple cube)
         and call modeflow.initialize()
      b) single step method consisting of
         - get results of the 1D model (passed though the queue) and prescribing it as a recharge on the top of the cube
         - call modflow.update()to perform next step
         - extract pressure heads from the top 
         - return them from the step so that Model3D could send them to 1D through the queue
   The test can use its own config.yaml and an approach applied in tests/model_1d. 
   It is a test only, but the Model3DAPI is a backbone of a new implementation so put all important bits to the implementation.
   Inject model building from the test since that would be test specific (simplistic geometry).
   See composed/modflow_driver_3d for inspiration, it has relatively good structure, but I want different order of steps in the time loop:
   (2.) = (1d model), 3. , 4. (modflow.update), 1. , 5. 
   It is also too wordy, try to make code more compact by deduplicaation and possibly creating attrs classes to pass data around.
   
   Milestone result: working test, no changes in the existing classes should be necessary, excercise writer as well
====   
   Now review the existing code in deep_model, this conains an older code that has its own entrypoints and lot of duplicit code. 
   However there are also several usefull test configurations in deep_model/config and very usefull plots and VTK output. 
   So I want to refactor key functionalities to the new backbone created in the milestones 1 up to 3.
====
4. Wrap the qgis_reader into a minimalistic class for: a) extracting geometry from the GIS project b) forming the modflow mesh with assigned materials.
   The geometry class should be able to be created from mere konfig file and the GIS, but then it should produce files in the workdir so that 
   the geometry class could be recreated just from theat folder now without the GIS. That should allow to separate GIS -> geometry step from actual simulation.
   Point is that we want to create geometry once on the desktop, check it i Paraview and then move to cluster and do remaining calculations there.
   The geometry should have no information about physical properties of the materials or which boundary conditions are applied.
   The geometry class should be abel to call a VTK geometry output function (should already be there) and functions to convert model XY coords to longitude, latitude and to JTSK (Křovák) back and forth.
   The command `hlavo/main.py build_model <config.yaml> <workdir>` used to work in man, but is now broken trying to simplify the Model3D convoluted classes. 
   Your goal is to make it work again but with clean implementation. Make the integration test `runs/composed_3D_only/run_build_model.sh` work again. 
   
5. Read the material properties from the config file into an xarray (variables for different vG paramters), material 'str' valued coord, (lo, init, hi) ... able to specify bounds for calibration.
   See the existing concept enabling prescribing each space dependent quantity a unique per material, but also provide defaults thorugh the `all` virtual material.
   Follow existing code to implement simulation build function it would take a common simulation config class, geometry class and material xarray dataset and it creates 
   the modflow simulation setup for the composed simulation (see the milestone 3.)
   
6. Implement class to represent a single pumping well (read from wells.zarr) able to appply the time dependent pumping to the model, could provide just basic
   representation and be passed to a method of the simulation build class. Idealy if that could be prescribed as a time series so that we just build the simulation and let it go.
   
7. Implement representation of the monitoring well and apply it as a reader to the interrupted modflow loop once after each modflow.update() ; in fact after the update the backend step is comppleted and we return to the Model3D time loop where we should call the backend for exctracting the wells values to write to ZARR throught the writer calss.
