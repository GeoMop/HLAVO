import flopy
import matplotlib.pyplot as plt

sim = flopy.mf6.MFSimulation.load(sim_name="os_experiment", sim_ws=".")
gwf = sim.get_model()              # or sim.get_model("modelname") if needed
grid = gwf.modelgrid

fig, ax = plt.subplots()
grid.plot(ax=ax)        # works for many grid types
ax.set_aspect("equal")
plt.show()