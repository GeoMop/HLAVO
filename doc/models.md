# Numerical Models

## Surface Soil Model

The surface model is based on ParFlow Richards flow configured in
`hlavo/soil_parflow/parflow_model.py` and documented in `hlavo/soil_parflow/README.md`.
It uses a 1D vertical column with Van Genuchten parameters and allows layered
permeability via piecewise regions. The top boundary is a flux condition
(`FluxConst`) driven by precipitation/ET, and the bottom uses a Dirichlet
equilibrium reference (`DirEquilRefPatch`). Typical test setups use two soil
layers with distinct hydraulic conductivities.

## Evapotranspiration

### Priestley–Taylor model

**Priestley–Taylor potential evapotranspiration** driven by an **energy proxy
constructed from downward shortwave (solar) radiation only**. In this core,
**longwave (thermal infrared) exchange is omitted**, so the available energy is
approximated from solar input and a prescribed albedo. Longwave effects are
added by the *Net radiation* submodel.

The Priestley–Taylor model reduces the number of empirical inputs compared to
the Penman–Monteith model by collapsing the aerodynamic (wind/VPD) contribution
into a single coefficient.

Output: potential evapotranspiration flux $f_{ET}$ (water-equivalent).  
Reference: [Priestley & Taylor (1972)](https://journals.ametsoc.org/view/journals/mwre/100/2/1520-0493_1972_100_0081_otaosh_2_3_co_2.xml)

#### Input variables
$T$ - air temperature at 2 m above ground $[K]$,
source_name: `air_temperature_2m`  

$E^{acc}_{\downarrow,SW}$ - downward shortwave (solar) radiant energy,
accumulated $[J\,m^{-2}]$,
source_name: `surface_solar_radiation_downwards`

#### Constants
$T_0 = 273.15$ - kelvin-to-celsius temperature offset $[K]$,
source_name: `cfg:T0_K`  

$\lambda_v = 2.45\times10^{6}$ - latent heat of vaporization of water
(near $20^\circ C$) $[J\,kg^{-1}]$,
source_name: `cfg:lambda_v_J_kg`  

$e_0 = 0.6108$ - saturation vapor pressure at $T_c=0^\circ C$ (water) $[kPa]$,
source_name: `cfg:svp_e0_kPa`  

$a_{svp} = 17.27$ - saturation vapor pressure coefficient (water) $[-]$,
source_name: `cfg:svp_a`  

$b_{svp} = 237.3$ - saturation vapor pressure coefficient (water) $[^\circ C]$,
source_name: `cfg:svp_b_C`  

$c_{svp} = 4098$ - saturation vapor pressure slope coefficient (water)
$[^\circ C]$,
source_name: `cfg:svp_c_C`  

$\gamma_0 = 0.067$ - reference psychrometric constant at standard pressure
$[kPa\,^\circ C^{-1}]$,
source_name: `cfg:gamma0_kPa_C`

**Psychrometric constant note:** $\gamma$ links temperature change to moisture
change in moist-air thermodynamics. In Penman-type formulas it weights the
**aerodynamic (drying power)** term relative to the **radiative (energy)** term.
A common definition is $\gamma = c_p P / (\epsilon \lambda_v)$, so it increases
with air pressure $P$ and decreases with latent heat $\lambda_v$.

#### Model parameters
$\alpha_{PT}$ - Priestley–Taylor coefficient multiplying the radiative term to
account for non-equilibrium and mild advection $[-]$,
source_name: `cfg:alpha_PT`,
suggested range: $[1.0,\,1.6]$ (larger under advective/drier boundary layers,
closer to 1 under near-equilibrium wet conditions).  

$\alpha$ - broadband surface albedo used in the shortwave-only net radiation
proxy $[-]$,
source_name: `cfg:albedo`,
suggested range: $[0.05,\,0.35]$ (low for dark/wet surfaces, high for bright
soil and snow).

#### Derived quantities
$T_c = T - T_0$ - air temperature in degrees Celsius $[^\circ C]$

$F_{\downarrow,SW} = \Delta E^{acc}_{\downarrow,SW}/\Delta t$ - downward
shortwave radiative flux $[W\,m^{-2}]$

$R_n \approx (1-\alpha)\,F_{\downarrow,SW}$ - **shortwave-only** net radiation
proxy (longwave terms omitted) $[W\,m^{-2}]$

$e_s(T_c)$ - saturation vapor pressure (water) $[kPa]$

$$
e_s(T_c)=e_0\,\exp\!\left(\frac{a_{svp}\,T_c}{T_c+b_{svp}}\right)
$$

$s_e$ - slope of the saturation vapor pressure curve at $T_c$
(i.e. $s_e = de_s/dT_c$) $[kPa\,^\circ C^{-1}]$

$$
s_e=\frac{c_{svp}\,e_s(T_c)}{(T_c+b_{svp})^2}
$$

$G_{rad}$ - radiative weighting factor $[-]$

$$
G_{rad}=\frac{s_e}{s_e+\gamma}
$$

**Meaning of $s_e$ and $G_{rad}$:** $e_s(T_c)$ increases nonlinearly with
temperature; $s_e$ quantifies how quickly the saturation vapor pressure changes
with temperature at current conditions. In Priestley–Taylor (and Penman-type
energy partitioning), $G_{rad}=s_e/(s_e+\gamma)$ is the fraction of available
energy that tends to go into latent heat (evaporation) rather than sensible heat
under near-equilibrium conditions.

#### Predicted quantities
$$
f_{ET}=\frac{F_{LE}}{\lambda_v} \quad
F_{LE}=\alpha_{PT}\,G_{rad}\,R_n \quad
G_{rad}=\frac{s_e}{s_e+\gamma} \quad
\gamma \approx \gamma_0
$$


### Net radiation

Adds **longwave (thermal infrared) exchange** and **surface thermal emission** to
obtain a more complete net radiation $R_n$ by combining shortwave and longwave
terms. Output: improved $R_n$ used by the Priestley–Taylor model.  
Reference: [Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

#### Input variables
$E^{acc}_{\downarrow,LW}$ - downward longwave radiant energy, accumulated
$[J\,m^{-2}]$,
source_name: `surface_thermal_radiation_downwards`  

$T_s$ - surface (skin) temperature $[K]$,
source_name: `surface_temperature`

#### Constants
$\sigma = 5.670374419\times10^{-8}$ - Stefan–Boltzmann constant
$[W\,m^{-2}\,K^{-4}]$,
source_name: `cfg:sigma_SB`

#### Model parameters
$\varepsilon$ - effective surface emissivity used for longwave emission $[-]$,
source_name: `cfg:emissivity`,
suggested range: $[0.92,\,0.99]$ (higher for vegetation/wet surfaces, lower for
dry mineral surfaces).  

$\alpha$ - broadband surface albedo $[-]$,
source_name: `cfg:albedo` (shared with Priestley–Taylor),
suggested range: $[0.05,\,0.35]$.

#### Derived quantities
$F_{\downarrow,LW}=\Delta E^{acc}_{\downarrow,LW}/\Delta t$ - downward longwave
radiative flux $[W\,m^{-2}]$

$F_{\uparrow,LW}=\varepsilon\,\sigma\,T_s^4$ - upward longwave radiative flux
(surface emission) $[W\,m^{-2}]$

$F_{\uparrow,SW}=\alpha\,F_{\downarrow,SW}$ - upward shortwave radiative flux
(surface reflection) $[W\,m^{-2}]$

#### Predicted quantity
$$
R_n=(F_{\downarrow,SW}-F_{\uparrow,SW})+(F_{\downarrow,LW}-F_{\uparrow,LW})
$$



### Penman–Monteith model (alternative core)

**Penman–Monteith potential evapotranspiration** combines (i) available energy
and (ii) aerodynamic drying power (wind + vapor pressure deficit) with explicit
surface and aerodynamic resistances. Output: potential evapotranspiration flux
$f_{ET}$ (water-equivalent).  
References: [Monteith (1965)](https://doi.org/10.1007/978-3-642-28647-0_3),
[Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

#### Input variables
$R_n$ - net radiation at the surface $[W\,m^{-2}]$,
source_name: `derived:R_n` (from *Net radiation* submodel)  

$RH$ - relative humidity at 2 m $[-]$,
source_name: `relative_humidity_2m`  

$u_{10}$ - wind speed at 10 m $[m\,s^{-1}]$,
source_name: `wind_speed_10m`  

$P_{sl}$ - air pressure at sea level $[Pa]$,
source_name: `air_pressure_at_sea_level`  

$z$ - site elevation above sea level $[m]$,
source_name: `cfg:site_elevation_m`

#### Constants
$p_0 = 1000$ - pascals per kilopascal $[Pa\,kPa^{-1}]$,
source_name: `cfg:Pa_per_kPa`  

$R_d = 287.05$ - specific gas constant for dry air $[J\,kg^{-1}\,K^{-1}]$,
source_name: `cfg:R_d_J_kgK`  

$c_p = 1004.67$ - specific heat of air at constant pressure $[J\,kg^{-1}\,K^{-1}]$,
source_name: `cfg:c_p_J_kgK`  

$\epsilon_w = 0.622$ - ratio of molecular weights (water vapor / dry air) $[-]$,
source_name: `cfg:epsilon_w`  

$g = 9.80665$ - gravitational acceleration $[m\,s^{-2}]$,
source_name: `cfg:g_m_s2`  

$\kappa = 0.41$ - von Kármán constant $[-]$,
source_name: `cfg:von_karman`  

$z_u = 10$ - wind measurement height $[m]$,
source_name: `cfg:z_u_m`  

$z_T = 2$ - temperature/humidity measurement height $[m]$,
source_name: `cfg:z_T_m`

#### Model parameters
$r_s$ - bulk surface (or canopy) resistance $[s\,m^{-1}]$,
source_name: `cfg:r_s_s_m`,
suggested range: $[30,\,300]$ (wet/active canopies low; dry soil/stressed
vegetation high).  

$h_c$ - effective canopy height $[m]$,
source_name: `cfg:canopy_height_m`,
suggested range: $[0.02,\,2]$ (grassland to shrubs; site-specific).  

$k_d$ - displacement height factor in $d=k_d h_c$ $[-]$,
source_name: `cfg:disp_factor`,
suggested range: $[0.5,\,0.8]$ (denser canopies higher).  

$k_{z0m}$ - roughness-length (momentum) factor in $z_{0m}=k_{z0m} h_c$ $[-]$,
source_name: `cfg:z0m_factor`,
suggested range: $[0.05,\,0.2]$.  

$k_{z0h}$ - roughness-length (heat/vapor) factor in $z_{0h}=k_{z0h} z_{0m}$ $[-]$,
source_name: `cfg:z0h_factor`,
suggested range: $[0.05,\,1]$ (often $\ll 1$).  

$f_G$ - soil heat flux fraction in $G=f_G R_n$ $[-]$,
source_name: `cfg:soil_heat_frac`,
suggested range: $[0,\,0.2]$ (daily steps near 0; sub-daily can be higher).

#### Derived quantities
$e_a = RH\,e_s(T_c)$ - actual vapor pressure $[kPa]$  

$VPD = e_s(T_c)-e_a$ - vapor pressure deficit $[kPa]$  

$P = P_{sl}\exp\!\left(-\frac{g z}{R_d T}\right)$ - site air pressure
(barometric approximation) $[Pa]$  

$\rho_a = P/(R_d T)$ - air density (ideal-gas approximation) $[kg\,m^{-3}]$  

$\Delta = p_0\,s_e$ - saturation-slope in $[Pa\,K^{-1}]$  

$D = p_0\,VPD$ - vapor pressure deficit in $[Pa]$  

$\gamma = \frac{c_p P}{\epsilon_w \lambda_v}$ - psychrometric constant
in $[Pa\,K^{-1}]$  

$d = k_d h_c$ - displacement height $[m]$  

$z_{0m} = k_{z0m} h_c$ - roughness length for momentum $[m]$  

$z_{0h} = k_{z0h} z_{0m}$ - roughness length for heat/vapor $[m]$  

$r_a = \frac{\ln\!\left(\frac{z_u-d}{z_{0m}}\right)\,
           \ln\!\left(\frac{z_T-d}{z_{0h}}\right)}
          {\kappa^2 u_{10}}$ - aerodynamic resistance $[s\,m^{-1}]$  

$G = f_G R_n$ - soil heat flux $[W\,m^{-2}]$

#### Predicted quantities
$$
F_{LE}=\frac{\Delta\,(R_n-G)+\rho_a c_p\,D/r_a}
             {\Delta+\gamma\left(1+r_s/r_a\right)} \quad
f_{ET}=\frac{F_{LE}}{\lambda_v} \\
$$


## Kalman filter and data asimilation
The surface model is wrapped in an Unscented Kalman Filter (UKF) using
`hlavo/kalman/kalman.py` with state/measurement schemas in
`hlavo/kalman/kalman_state.py`. Measurements are soil moisture profiles; the UKF
updates states and selected parameters (e.g., van Genuchten values) and can run
either from recorded measurements or from synthetic forward runs. Parallel
sigma-point propagation uses thread-based UKF in `hlavo/kalman/parallel_ukf.py`.

## Deep Model
The deep vadose zone model is generated from GIS inputs and exported as MODFLOW 6
inputs via `hlavo/deep_model/qgis_reader.py` and
`hlavo/deep_model/build_modflow_grid.py`. Configuration is in
`hlavo/deep_model/model_config.yaml`, and the workflow is described in
`hlavo/deep_model/README.md`. The model construction builds a grid, assigns
materials by layer surfaces, and runs a steady-state Darcy flow with no-flow
sides/bottom and a seepage/recharge condition on top.

## Coupling
Coupling between surface and deep models is planned as a parallel workflow with
`dask.distributed`, exchanging boundary fluxes and pressure heads through queues.
The current mock implementation is in `hlavo/composed/composed_model_mock.py`,
with distributed execution utilities in `hlavo/composed/run_map.py`. 
The surface model predicts flux to the top boundary of the deep model, while the deep model 
provides preasure head to at the bottom boundary of the surface models.
The main script is `hlavo/composed/main.py`.
