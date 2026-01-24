# ET inputs and formulas (Priestley–Taylor + extensions)

Notation uses compact symbols mapped to **verbatim source variable names** in
backticks. For extensions, **only newly added variables** are listed.

All water fluxes are denoted by $f_{\bullet}$ and use units $[mm\,s^{-1}]$
unless stated otherwise.

---

## Priestley–Taylor model

### Model overview
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

### Input variables
$T$ - air temperature at 2 m above ground $[K]$,
source_name: `air_temperature_2m`  

$E^{acc}_{\downarrow,SW}$ - downward shortwave (solar) radiant energy,
accumulated $[J\,m^{-2}]$,
source_name: `surface_solar_radiation_downwards`

### New constants
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

### New parameters
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

### Derived quantities
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

### Model
$$
f_{ET}=\frac{F_{LE}}{\lambda_v} \quad
F_{LE}=\alpha_{PT}\,G_{rad}\,R_n \quad
G_{rad}=\frac{s_e}{s_e+\gamma} \quad
\gamma \approx \gamma_0
$$

---

## Net radiation

### Model overview
Adds **longwave (thermal infrared) exchange** and **surface thermal emission** to
obtain a more complete net radiation $R_n$ by combining shortwave and longwave
terms. Output: improved $R_n$ used by the Priestley–Taylor model.  
Reference: [Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables (new)
$E^{acc}_{\downarrow,LW}$ - downward longwave radiant energy, accumulated
$[J\,m^{-2}]$,
source_name: `surface_thermal_radiation_downwards`  

$T_s$ - surface (skin) temperature $[K]$,
source_name: `surface_temperature`

### New constants
$\sigma = 5.670374419\times10^{-8}$ - Stefan–Boltzmann constant
$[W\,m^{-2}\,K^{-4}]$,
source_name: `cfg:sigma_SB`

### New parameters
$\varepsilon$ - effective surface emissivity used for longwave emission $[-]$,
source_name: `cfg:emissivity`,
suggested range: $[0.92,\,0.99]$ (higher for vegetation/wet surfaces, lower for
dry mineral surfaces).  

$\alpha$ - broadband surface albedo $[-]$,
source_name: `cfg:albedo` (shared with Priestley–Taylor),
suggested range: $[0.05,\,0.35]$.

### Derived quantities
$F_{\downarrow,LW}=\Delta E^{acc}_{\downarrow,LW}/\Delta t$ - downward longwave
radiative flux $[W\,m^{-2}]$

$F_{\uparrow,LW}=\varepsilon\,\sigma\,T_s^4$ - upward longwave radiative flux
(surface emission) $[W\,m^{-2}]$

$F_{\uparrow,SW}=\alpha\,F_{\downarrow,SW}$ - upward shortwave radiative flux
(surface reflection) $[W\,m^{-2}]$

### Model
$$
R_n=(F_{\downarrow,SW}-F_{\uparrow,SW})+(F_{\downarrow,LW}-F_{\uparrow,LW})
$$

---

## Drying power (optional diagnostics)

### Model overview
Computes atmospheric diagnostics that quantify air-side drying demand: vapor
pressure deficit $VPD$, pressure-based psychrometric constant $\gamma(P)$, and
wind speed $u_{10}$. These diagnostics are **not required** by the
Priestley–Taylor core but are useful for quality assurance and for later
Penman–Monteith implementation (or optional empirical adjustments of
$\alpha_{PT}$).  
Reference: [Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables (new)
$RH$ - relative humidity at 2 m $[-]$,
source_name: `relative_humidity_2m`  

$u_{10}$ - wind speed at 10 m $[m\,s^{-1}]$,
source_name: `wind_speed_10m`  

$P_{sl}$ - air pressure at sea level $[Pa]$,
source_name: `air_pressure_at_sea_level`

### New constants
$p_0 = 1000$ - pascals per kilopascal $[Pa\,kPa^{-1}]$,
source_name: `cfg:Pa_per_kPa`

### New parameters
$k_\gamma$ - proportionality for psychrometric constant $\gamma = k_\gamma P$
$[kPa\,^\circ C^{-1}\,kPa^{-1}]$,
source_name: `cfg:k_gamma`,
suggested range: within $\pm 10\%$ of the FAO-56 value (uncertainty increases if
$P_{sl}$ is used as a proxy for local pressure without elevation correction).  

$f_{\alpha}(\cdot)$ - optional modifier of $\alpha_{PT}$ based on $u_{10}$ and
$VPD$ $[-]$,
source_name: `cfg:alphaPT_modifier`,
suggested range: close to 1 unless explicitly calibrated for strong advection.

### Derived quantities
$P=P_{sl}/p_0$ - pressure in kilopascals $[kPa]$

$e_a=RH\,e_s(T_c)$ - actual vapor pressure $[kPa]$

$VPD=e_s(T_c)-e_a$ - vapor pressure deficit $[kPa]$

### Model
$$
P=\frac{P_{sl}}{p_0} \quad
\gamma=k_\gamma\,P \quad
e_a=RH\,e_s(T_c) \quad
VPD=e_s(T_c)-e_a \quad
\alpha_{PT}^{eff}=f_{\alpha}(u_{10},VPD)\,\alpha_{PT}\ (\text{optional})
$$

---

## Penman–Monteith model (alternative core)

### Model overview
**Penman–Monteith potential evapotranspiration** combines (i) available energy
and (ii) aerodynamic drying power (wind + vapor pressure deficit) with explicit
surface and aerodynamic resistances. Output: potential evapotranspiration flux
$f_{ET}$ (water-equivalent).  
References: [Monteith (1965)](https://doi.org/10.1007/978-3-642-28647-0_3),
[Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables
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

### New constants
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

### New parameters
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

### Derived quantities (new)
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

### Model
$$
F_{LE}=\frac{\Delta\,(R_n-G)+\rho_a c_p\,D/r_a}
             {\Delta+\gamma\left(1+r_s/r_a\right)} \quad
f_{ET}=\frac{F_{LE}}{\lambda_v} \\
e_a=RH\,e_s(T_c) \quad
VPD=e_s(T_c)-e_a \quad
P=P_{sl}\exp\!\left(-\frac{g z}{R_d T}\right) \quad
\rho_a=\frac{P}{R_d T} \\
\Delta=p_0 s_e \quad
D=p_0 VPD \quad
\gamma=\frac{c_p P}{\epsilon_w \lambda_v} \quad
G=f_G R_n \\
d=k_d h_c \quad
z_{0m}=k_{z0m} h_c \quad
z_{0h}=k_{z0h} z_{0m} \quad
r_a=\frac{\ln\!\left(\frac{z_u-d}{z_{0m}}\right)\,
         \ln\!\left(\frac{z_T-d}{z_{0h}}\right)}
        {\kappa^2 u_{10}}
$$

---

# Further extensions (lumped ODE submodels)

All submodels below introduce **state variables** and produce modified fluxes
for Richards upper boundary and root uptake. Each section lists new inputs only.

---

## Snowpack ODE (storage + melt release)

### Model overview
Lumped snow water store that delays liquid water input until melt. Outputs the
liquid water flux to the ground surface $f_{P,\ell}$.  
Reference (degree-day snowmelt lineage):  
[Anderson (1973)](https://repository.library.noaa.gov/view/noaa/13507/noaa_13507_DS1.pdf)

### Input variables (new)
$P_r^{acc}$ - rainfall amount, accumulated $[kg\,m^{-2}]$,
source_name: `rainfall_amount_accum`  
$P_s^{acc}$ - snowfall amount, accumulated $[kg\,m^{-2}]$,
source_name: `snowfall_amount_accum`

### New constants
$\rho_w$ - water-equivalent conversion factor $[mm\,(kg\,m^{-2})^{-1}]$,
source_name: `cfg:rho_w_mm_per_kgm2`  
$T_0$ - melt threshold temperature $[K]$, source_name: `cfg:T0_K`

### New parameters
$c_m$ - degree-day melt factor $[mm\,s^{-1}\,K^{-1}]$,
source_name: `cfg:melt_factor`,
suggested range: corresponding to roughly $0.1$–$10$ mm/day/K (higher with strong
radiation/advection, lower in shaded/calm conditions).

### State variables
$S(t)$ - snow water equivalent store on ground $[mm]$

### Derived quantities
$f_{P,r}=\rho_w\,\Delta P_r^{acc}/\Delta t$ - rainfall rate $[mm\,s^{-1}]$  
$f_{P,s}=\rho_w\,\Delta P_s^{acc}/\Delta t$ - snowfall rate $[mm\,s^{-1}]$

### Model
$$
f_{P,r}=\rho_w\,\frac{\Delta P_r^{acc}}{\Delta t} \quad
f_{P,s}=\rho_w\,\frac{\Delta P_s^{acc}}{\Delta t} \quad
f_m=c_m\,\max(T-T_0,0) \\
f_m\leftarrow \min\!\left(f_m,\frac{S}{\Delta t}+f_{P,s}\right) \quad
\frac{dS}{dt}=f_{P,s}-f_m \quad
f_{P,\ell}=f_{P,r}+f_m
$$

---

## Frozen-soil gate ODE (effective permeability)

### Model overview
Lumped freeze/thaw index that reduces effective liquid input to the soil surface
when the surface is frozen, in addition to any infiltration limitation emerging
from the Richards/seepage-face boundary. Outputs $f_{P,\ell,eff}$.  
Reference (frozen-soil infiltration effects):  
[Gray et al. (1985)](https://research-groups.usask.ca/hydrology/documents/pubs/papers/gray_et_al_1985_3.pdf)

### Input variables (new)
$T_s$ - surface (skin) temperature $[K]$, source_name: `surface_temperature`

### New constants
$T_0$ - freezing reference temperature $[K]$, source_name: `cfg:T0_K`

### New parameters
$\Delta T_f$ - freeze transition width $[K]$,
source_name: `cfg:freeze_width_K`,
suggested range: $[0.5,\,5]$ K (controls how “binary” freezing appears).  

$\tau_f$ - freezing time scale $[s]$, source_name: `cfg:tau_freeze_s`,
suggested range: hours to about a week (site-dependent).  

$\tau_t$ - thawing time scale $[s]$, source_name: `cfg:tau_thaw_s`,
suggested range: hours to about a week (site-dependent).  

$p_f$ - sharpness exponent in permeability multiplier $[-]$,
source_name: `cfg:freeze_exponent`,
suggested range: $[1,\,8]$ (higher makes permeability collapse near full freeze).

### State variables
$F(t)$ - frozen fraction / ice-blockage index $[-]$ (0–1)

### Derived quantities
$F^*(T_s)$ - equilibrium frozen fraction $[-]$  
$g_F=(1-F)^{p_f}$ - permeability multiplier $[-]$

### Model
$$
F^*(T_s)=\mathrm{clip}\!\left(\frac{T_0-T_s}{\Delta T_f},0,1\right) \quad
g_F=(1-F)^{p_f} \\
\frac{dF}{dt}=
\begin{cases}
\frac{F^*(T_s)-F}{\tau_f}, & F^*(T_s)>F \\
\frac{F^*(T_s)-F}{\tau_t}, & F^*(T_s)\le F
\end{cases}
\quad
f_{P,\ell,eff}=g_F\,f_{P,\ell}
$$

---

## Canopy interception ODE (rain and snow storage + drip)

### Model overview
Lumped canopy liquid-water and canopy snow stores that delay delivery to the
ground and allow wet-canopy losses. Outputs modified ground inputs $f_{P,\ell}$
(liquid) and $f_{P,s}$ (snow).  
Reference (canopy storage model):  
[Rutter et al. (1971)](https://www.sciencedirect.com/science/article/pii/0002157171900343)

### Input variables (new)
(uses $f_{P,r}$, $f_{P,s}$, $T$ from earlier sections)

### New parameters
$C_r$ - canopy liquid storage capacity $[mm]$,
source_name: `cfg:canopy_Cr_mm`,
suggested range: $[0.1,\,2]$ mm (depends on vegetation structure and leaf area).  

$C_s$ - canopy snow storage capacity $[mm]$,
source_name: `cfg:canopy_Cs_mm`,
suggested range: $[0.5,\,10]$ mm (depends on canopy type and snow loading).  

$\tau_r$ - drip time scale for canopy liquid water $[s]$,
source_name: `cfg:canopy_tau_r_s`,
suggested range: minutes to about a day (wind and structure).  

$\tau_s$ - unloading time scale for canopy snow $[s]$,
source_name: `cfg:canopy_tau_s_s`,
suggested range: hours to about a week (snow cohesion and wind).  

$c_{m,c}$ - canopy snow melt factor $[mm\,s^{-1}\,K^{-1}]$,
source_name: `cfg:canopy_melt_factor`,
suggested range: comparable order to $c_m$ (more exposed canopies melt faster).  

$k_i$ - wet-canopy evaporation rate constant $[s^{-1}]$,
source_name: `cfg:wet_canopy_evap_rate`,
suggested range: $[10^{-7},\,10^{-4}]$ s$^{-1}$ (controls strength of wet-canopy
losses; influenced by wind and radiation).

### State variables
$W_r(t)$ - canopy liquid water store $[mm]$  
$W_s(t)$ - canopy snow store $[mm]$

### Derived quantities
$f_{I,r},f_{TF,r}$ - intercepted rain and throughfall $[mm\,s^{-1}]$  
$f_{I,s},f_{TF,s}$ - intercepted snow and throughfall $[mm\,s^{-1}]$  
$f_{D,r}$ - canopy drip (liquid) $[mm\,s^{-1}]$  
$f_{U,s}$ - canopy snow unloading $[mm\,s^{-1}]$  
$f_{M,cs}$ - canopy snow melt to liquid $[mm\,s^{-1}]$  
$f_{E,i}$ - wet-canopy evaporation loss $[mm\,s^{-1}]$

### Model
$$
f_{I,r}=\min\!\left(f_{P,r},\max\!\left(0,\frac{C_r-W_r}{\Delta t}\right)\right) \quad
f_{TF,r}=f_{P,r}-f_{I,r} \quad
f_{D,r}=\frac{W_r}{\tau_r} \quad
f_{E,i}=k_i\,W_r \\
f_{I,s}=\min\!\left(f_{P,s},\max\!\left(0,\frac{C_s-W_s}{\Delta t}\right)\right) \quad
f_{TF,s}=f_{P,s}-f_{I,s} \quad
f_{U,s}=\frac{W_s}{\tau_s} \quad
f_{M,cs}=c_{m,c}\,\max(T-T_0,0) \\
f_{M,cs}\leftarrow \min\!\left(f_{M,cs},\frac{W_s}{\Delta t}\right) \quad
\frac{dW_r}{dt}=f_{I,r}-f_{E,i}-f_{D,r} \quad
\frac{dW_s}{dt}=f_{I,s}-f_{M,cs}-f_{U,s} \\
f_{P,\ell}=f_{TF,r}+f_{D,r}+f_{M,cs}+f_m \quad
f_{P,s}=f_{TF,s}+f_{U,s}
$$

---

## Vegetation stress ODE (transpiration demand limiter)

### Model overview
Lumped root-zone “available water” index that reduces transpiration demand before
it is applied as Richards root uptake. Output is stressed transpiration demand
$f_{T,a}$.  
Reference (macroscopic root water uptake and stress concepts):  
[Skaggs et al. (2006)](https://www.pc-progress.com/Documents/RVGenugten/Skaggs_Root_water_uptake_AWM.pdf)

### Input variables (new)
(uses $f_{ET}$ from PT core; uses $f_{P,\ell,eff}$ from frozen-soil gating)

### New parameters
$\theta_w$ - wilting threshold of the root-zone index $[mm]$,
source_name: `cfg:theta_w_mm`,
suggested range: site dependent (order 10–200 mm based on rooting depth and
plant-available water).  

$\theta_{fc}$ - non-stress threshold (field-capacity proxy) $[mm]$,
source_name: `cfg:theta_fc_mm`,
suggested range: site dependent (order 10–200 mm; must satisfy
$\theta_{fc}>\theta_w$).  

$f_T$ - potential transpiration fraction of $f_{ET}$ $[-]$,
source_name: `cfg:f_T`,
suggested range: $[0.1,\,0.9]$ (same interpretation as in Partitioning).  

$a_r$ - fraction of surface liquid input contributing to root-zone index within
about a week $[-]$, source_name: `cfg:root_recharge_fraction`,
suggested range: $[0,\,1]$ (smaller if water bypasses roots, larger for shallow
roots and strong capillary connection).  

$b_r$ - leak/drain rate from the index $[s^{-1}]$,
source_name: `cfg:root_leak_rate`,
suggested range: week-scale, i.e. on the order of $10^{-7}$–$10^{-5}$ s$^{-1}$
(higher drains faster).

### State variables
$\theta_r(t)$ - root-zone available water index $[mm]$

### Derived quantities
$f_s(\theta_r)$ - stress factor $[-]$  
$f_{T,p}=f_T\,f_{ET}$ - potential transpiration demand $[mm\,s^{-1}]$  
$f_{T,a}=f_s(\theta_r)\,f_{T,p}$ - stressed transpiration demand $[mm\,s^{-1}]$

### Model
$$
f_s(\theta_r)=\mathrm{clip}\!\left(
\frac{\theta_r-\theta_w}{\theta_{fc}-\theta_w},0,1\right) \quad
f_{T,p}=f_T\,f_{ET} \quad
f_{T,a}=f_s(\theta_r)\,f_{T,p} \\
\frac{d\theta_r}{dt}=a_r\,f_{P,\ell,eff}-f_{T,a}-b_r\,\theta_r
$$

---

## Summary of new ODE states (all extensions)
- Snowpack store: $S(t)$  
- Frozen gate: $F(t)$  
- Canopy stores: $W_r(t)$, $W_s(t)$  
- Vegetation stress index: $\theta_r(t)$

These states produce modified coupling fluxes:
- liquid input to soil: $f_{P,\ell,eff}$  
- transpiration sink demand: $f_{T,a}$
