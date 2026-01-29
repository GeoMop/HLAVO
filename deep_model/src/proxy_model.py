# ## OS: Attempt of so-called "deep model" for HLAVO project.
# ## Derived form "Lake Package Problem 1" provided by flopy package
#

# ### Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# +
from pathlib import Path   

import flopy
import matplotlib.pyplot as plt
import numpy as np
from flopy.plot.styles import styles
from modflow_devtools.misc import get_env, timed

# ====================================================================================


# OS: the simple setup, just store everything in current working directory  
sim_name = "os_experiment"
workspace = Path.cwd()
figs_path = Path.cwd()

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "meters" # "feets" or "meters"
time_units = "days"

# Model parameters
nper = 1  # Number of periods. OS: Period = one setup for BCD, well pumping rate etc.
nlay = 16  # Number of layers. OS: Very thin layers on the top, then thicker 
nrow = 12  # Number of rows. OS: 10 rows for actual model, one row on each side for BCD 
ncol = 12  # Number of columns. OS: dtto 
top =  100.0  # Top of the model ($m a.s.l.$) 
# OS: Bottoms of layers ($m a.s.l.$)
botm = [ 98., 96., 94., 92., 88., 84., 80., 72., 64., 56., 48., 40., 30., 20., 10., 0. ] # Bottom elevations ($ft$)
strt = 50.0  # Starting head ($m$) OS: measured from where? m.a.s.l.? TODO find out!
k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$) OS: purposefuly unrealistic
k33 = 1.0  # Vertical hydraulic conductivity ($m/d$) OS: purposefuly unrealistic
ss = 3e-4  # Specific storage ($1/d$) OS: From the "lake" example, not changed 
sy = 0.2  # Specific yield (unitless) OS: From the "lake" example, not changed
H1 = 110.0 # Constant head on left side of model ($m$) OS: The same remark as for "strt" above
H2 = 60.0 # Constant head on right side of model ($m$) OS: dtto

# OS: Important, but not used in this model. We need recharge as an array
# recharge = 0.0116  # Aereal recharge rate ($m/d$) OS: unit: $m/d = $ ( m^3 / d ) / m^2 $

# OS: Not used in this model
# etvrate = 0.0141  # Maximum evapotranspiration rate ($ft/d$)
# etvdepth = 15.0  # Evapotranspiration extinction depth ($ft$)

# Static temporal data used by TDIS file
# TODO tune
# OS: time discretization: length of a stress period, number of timesteps, multiplier for the length of timestep
tdis_ds = ((5000.0, 100, 1.0),) 

# OS: size of cells in x-direction ( = rows)
# define delr and delc
delr = np.array(
    [
        2.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        2.0,
    ]
)
# OS: size of cells in y-direction ( = columns)
delc = np.array(
    [
        2.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        2.0,
    ]
)

# Define dimensions
# OS: Kept unchanged, used only for vizualisation
extents = (0.0, delr.sum(), 0.0, delc.sum())
shape2d = (nrow, ncol)
shape3d = (nlay, nrow, ncol)

# OS: Lake, we don't want one in our model
# Create the array defining the lake location
# lake_map = np.ones(shape3d, dtype=np.int32) * -1
# lake_map[0, 6:11, 6:11] = 0
# lake_map[1, 7:10, 7:10] = 0
# lake_map = np.ma.masked_where(lake_map < 0, lake_map)

# OS: This could be handy one day
# create linearly varying evapotranspiration surface
# xlen = delr.sum() - 0.5 * (delr[0] + delr[-1])
# x = 0.0
# s1d = H1 * np.ones(ncol, dtype=float)
# for idx in range(1, ncol):
#     x += 0.5 * (delr[idx - 1] + delr[idx])
#     frac = x / xlen
#     s1d[idx] = H1 + (H2 - H1) * frac
# surf = np.tile(s1d, (nrow, 1))
# surf[lake_map[0] == 0] = botm[0] - 2
# surf[lake_map[1] == 0] = botm[1] - 2

# OS: Technique for setup of the BCDs
# Constant head boundary conditions
chd_spd = []
for k in range(nlay):
    chd_spd += [[k, i, 0, H1] for i in range(nrow)]
    # chd_spd += [[k, i, ncol - 1, H2] for i in range(nrow)]

# OS: IMPORTANT - "drain" boundary condition, north face of our 
# model = face of the quarry, open to atmosphere
# Drain boundary condition
drn_spd = []
for k in range(nlay):
    # OS: Not sure what the numbers are yet. layer, row, column, elevation, conductance
    drn_spd += [[k, 0 , i , 100, 100 ] for i in range(ncol)]
# print( drn_spd )

# OS: Not changed
# Solver parameters
nouter = 500
ninner = 100
hclose = 1e-9
rclose = 1e-6

# OS: Fow vizualisation 
# Figure properties
figure_size = (6.3, 5.6)
masked_values = (0, 1e30, -1e30)


# ====================================================================================

def build_models( recharge = 0.01 ):
    sim_ws = workspace / sim_name
    # OS: Main simulation object
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")
    # OS: And now the packages
    # OS: Time discretization
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
    # OS: Solver, no idea what the parameters mean yet.
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        linear_acceleration="bicgstab",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=f"{rclose} strict",
    )
    # OS: Groundwater flow package
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=sim_name, newtonoptions="newton", save_flows=True
    )
    # OS: Discretization for groundwater flow
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        idomain=np.ones(shape3d, dtype=int),
        top=top,
        botm=botm,
    )
    # OS: npf = node property flow
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=k11,
        k33=k33,
        save_specific_discharge=True,
    )
    # OS: Storativity and company...
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=1,
        sy=sy,
        ss=ss,
    )
    # OS: Initial conditions
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    # OS: Constant head ( = Dirichlet's) BCD
    # flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    # OS: Recharge 
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    # OS: Drainage
    flopy.mf6.ModflowGwfdrn( gwf, drn_spd )

    head_filerecord = f"{sim_name}.hds"
    budget_filerecord = f"{sim_name}.cbc"
    # OS: Output control
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    return sim

# ====================================================================================

def write_models(sim, silent=True):
    sim.write_simulation(silent=False)
# ====================================================================================


@timed
def run_models(sim, silent=True):
    success, buff = sim.run_simulation(silent=False)
    assert success, buff

# ====================================================================================
# OS: copied form lake example, could be useful

def plot_grid(gwf, silent=True):
    # load the observations
    # lak_results = gwf.lak.output.obs().data

    # create MODFLOW 6 head object
    hobj = gwf.output.head()

    # create MODFLOW 6 cell-by-cell budget object
    cobj = gwf.output.budget()

    kstpkper = hobj.get_kstpkper()

    head = hobj.get_data(kstpkper=kstpkper[0])
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
        cobj.get_data(text="DATA-SPDIS", kstpkper=kstpkper[0])[0], gwf
    )

    with styles.USGSMap():
        fig = plt.figure(figsize=(4, 6.9), tight_layout=True)
        plt.axis("off")

        nrows, ncols = 10, 1
        axes = [fig.add_subplot(nrows, ncols, (1, 5))]
        axes.append(fig.add_subplot(nrows, ncols, (6, 8), sharex=axes[0]))

        for idx, ax in enumerate(axes):
            ax.set_xlim(extents[:2])
            if idx == 0:
                ax.set_ylim(extents[2:])
                ax.set_aspect("equal")

        # legend axis
        axes.append(fig.add_subplot(nrows, ncols, (9, 10)))

        # set limits for legend area
        ax = axes[-1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # get rid of ticks and spines for legend area
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.patch.set_alpha(0.0)

        ax = axes[0]
        mm = flopy.plot.PlotMapView(gwf, ax=ax, extent=extents)
        # mm.plot_bc("CHD", color="cyan")
        mm.plot_inactive(color_noflow="#5DBB63")
        mm.plot_grid(lw=0.5, color="black")
        cv = mm.contour_array(
            head,
            levels=np.arange(140, 160, 2),
            linewidths=0.75,
            linestyles="-",
            colors="blue",
            masked_values=masked_values,
        )
        plt.clabel(cv, fmt="%1.0f")
        mm.plot_vector(qx, qy, normalize=True, color="0.75")
        ax.set_xlabel("x-coordinate, in feet")
        ax.set_ylabel("y-coordinate, in feet")
        styles.heading(ax, heading="Map view", idx=0)
        styles.remove_edge_ticks(ax)

        ax = axes[1]
        xs = flopy.plot.PlotCrossSection(gwf, ax=ax, line={"row": 8})
        xs.plot_array(np.ones(shape3d), head=head, cmap="jet")
        # xs.plot_bc("CHD", color="cyan", head=head)
        # xs.plot_ibound(color_noflow="#5DBB63", head=head)
        xs.plot_grid(lw=0.5, color="black")
        ax.set_xlabel("x-coordinate")
        ax.set_ylim(0 , 100 )
        # ax.set_ylim(67, 160)
        ax.set_ylabel("Elevation")
        styles.heading(ax, heading="Cross-section view", idx=1)
        styles.remove_edge_ticks(ax)

        # legend
        ax = axes[-1]
   
        ax.plot(
            -10000,
            -10000,
            lw=0,
            marker="s",
            ms=10,
            mfc="cyan",
            mec="black",
            markeredgewidth=0.5,
            label="Constant-head boundary",
        )
        ax.plot(
            -10000,
            -10000,
            lw=0,
            marker="s",
            ms=10,
            mfc="blue",
            mec="black",
            markeredgewidth=0.5,
            label="Water table",
        )
 
        ax.plot(
            -10000, -10000, lw=0.75, ls="-", color="blue", label=r"Head contour, $ft$"
        )
        ax.plot(
            -10000,
            -10000,
            lw=0,
            marker="$\u2192$",
            ms=10,
            mfc="0.75",
            mec="0.75",
            label="Normalized specific discharge",
        )
        styles.graph_legend(ax, loc="lower center", ncol=2)

        plt.show()
        fpth = figs_path / f"{sim_name}-grid.png"
        # fig.savefig(fpth)

# ====================================================================================

# def plot_lak_results(gwf, silent=True):
#     with styles.USGSPlot():
#         # load the observations
#         lak_results = gwf.lak.output.obs().data
#         gwf_results = gwf.obs[0].output.obs().data

#         dtype = [
#             ("time", float),
#             ("STAGE", float),
#             ("A", float),
#             ("B", float),
#         ]

#         results = np.zeros((lak_results.shape[0] + 1), dtype=dtype)
#         results["time"][1:] = lak_results["totim"]
#         results["STAGE"][0] = 110.0
#         results["STAGE"][1:] = lak_results["STAGE"]
#         results["A"][0] = 115.0
#         results["A"][1:] = gwf_results["A"]
#         results["B"][0] = 115.0
#         results["B"][1:] = gwf_results["B"]

#         # create the figure
#         fig, ax = plt.subplots(
#             ncols=1, nrows=1, sharex=True, figsize=(6.3, 3.15), constrained_layout=True
#         )

#         ax.set_xlim(0, 3000)
#         ax.set_ylim(110, 160)
#         ax.plot(
#             results["time"],
#             results["STAGE"],
#             lw=0.75,
#             ls="--",
#             color="black",
#             label="Lake stage",
#         )
#         ax.plot(
#             results["time"], results["A"], lw=0.75, ls="-", color="0.5", label="Point A"
#         )
#         ax.plot(
#             results["time"],
#             results["B"],
#             lw=0.75,
#             ls="-",
#             color="black",
#             label="Point B",
#         )
#         ax.set_xlabel("Simulation time, in days")
#         ax.set_ylabel("Head or stage, in feet")
#         styles.graph_legend(ax, loc="lower right")

#         if plot_show:
#             plt.show()
#         if plot_save:
#             fpth = figs_path / f"{sim_name}-01.png"
#             fig.savefig(fpth)

# ====================================================================================

def plot_results(sim, silent=True):
    gwf = sim.get_model(sim_name)
    plot_grid(gwf, silent=silent)
    # plot_lak_results(gwf, silent=silent)

# ====================================================================================
# OS: which well recharge goes to specified surface cell:

def well_area( i, j ):
    if i <= 5 and j <= 5:
        return 0
    elif i <= 5:
        return 1
    elif j <= 5:
        return 2
    else:
        return 3
# ====================================================================================
# OS: Completely mine

if __name__ == '__main__':
    # recharge - input from Kalman
    # proxy model - assuming 4 wells, located in cells (3,3), (3,8), (8,3), (8,8)
    # and recharge on the surface spread acordingly 
    kwells = [ 0.01, 0.02, 0.03, 0.04 ]
    rch_spd = []
    for i in range( nrow ):
        for j in range( ncol ):
            rch_spd += [kwells[ well_area( i, j) ]]
    # This will be input of this model: An array of size nrow * ncol containing 
    # recharge rates for each cell on the surface layer of the grid 
    # Here it is proxied by the rch_spd[] array

    sim = build_models( recharge=rch_spd )
    write_models(sim)
    run_models(sim)

    # Now the output - heads on the top layer 
    gwf = sim.get_model(sim_name)
    hobj = gwf.output.head()
    kstpkper = hobj.get_kstpkper()
    head = hobj.get_data(kstpkper=kstpkper[0])
    # This array will go back to Kalman model
    shead = head[ 0, : , : ]
    # Here just print it
    print( shead )
    plot_results(sim)

# -