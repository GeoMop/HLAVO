Prompting:
Fix the "composed_model_mock.py" to run, but only fix the bugs, no radical changes in the code.

Add 3D model configuration in config file - Name, folder, time step
Add handling of Modflow model in a way that is in just_run.py
Start of the composed run: 3DModel sends to each 1D model through its queue object of the data3dto1D some initial value, by which starts the 1D models.
Then the main loop of the calculation:
3D model gets the recharges from 1D models through the queue. If it is the first iteration, 3D model does an assignment of the cells to 1d models - see below.
3D model spreads the recharges obtained form 1D models through upper layer of the 3D model's grid and writes a Modflow's RCHA file. Sperading the recharges see below.
3D model runs mf6 with given timestep.
After succesful calculation, heads at the end of the timestep are written as an initial condition for the next step.
Head for each 1D model is calculated and send to this 1D model through its queue - calculations of the heads see below.
End of main loop

assignment:
3D model has geographical locations (lon/lan) of all 1D models.
3D model knows gographical coordinates (lon/lan) of SW and NE corners of the grid - they are written in the nam file
Calculate positions of the 1D models in the grid 
Assign each cell of the top layer of the 3d model to one 1d model - the nearest one (Voronoi algorithm)

spreading the recharges:
data from each 1d model (recharge) for given timestep are set to all cells of the 3D model assigned to that 1d model

calculation of the heads:
Calculate average hydraulic head from all cells belonging to the particular 1D model
send this head to that 1d model