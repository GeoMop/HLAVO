# ET inputs and formulas (Priestley–Taylor + extensions)

Notation uses compact symbols mapped to **verbatim source variable names** in
backticks. For extensions, **only newly added variables** are listed.

---

## PT core

### Model overview
**Priestley–Taylor potential ET** from an energy proxy (shortwave only [TODO: explain better what you mean by shortwave only probably lack of infrared model]). Outputs
potential evapotranspiration flux $ET_p$ [TODO: rename as $f_{ET}$ using later $f$ for various water fluxes further on] as water-equivalent rate.  
Reference: [Priestley & Taylor (1972)](https://journals.ametsoc.org/view/journals/mwre/100/2/1520-0493_1972_100_0081_otaosh_2_3_co_2.xml)

### Input variables
$T$ - 2 m air temperature $[K]$, source_name: `air_temperature_2m`  
$S^{acc}_{\downarrow}$ - downw.[TODO: for whole document use full words, no shortcuts] shortwave, accumulated $[J\,m^{-2}]$,
source_name: `surface_solar_radiation_downwards`

### New constants
$T_0 = 273.15$ - K-to-°C offset $[K]$, source_name: `cfg:T0_K`  
$e_0 = 0.6108$ - sat. vap. pressure at $0^\circ C$ $[kPa]$,
source_name: `cfg:e0_kPa`  
$a = 17.27$ - sat. vap. pressure shape [TODO: specify if a, b, c parameters are for water] $[-]$, source_name: `cfg:svp_a`  
$b = 237.3$ - sat. vap. pressure shape $[^\circ C]$, source_name: `cfg:svp_b_C`  
$c = 4098$ - slope coefficient $[^\circ C]$, source_name: `cfg:svp_c_C`  
$\gamma_0 = 0.067$ - reference psychrometric const. $[kPa\,^\circ C^{-1}]$,
source_name: `cfg:gamma0_kPa_C`  
$\lambda_v = 2.45\times10^6$ - latent heat (water) $[J\,kg^{-1}]$,
source_name: `cfg:lambda_v_J_kg`

### New parameters
$\alpha_{PT}$ - Priestley–Taylor coefficient $[-]$, source_name: `cfg:alpha_PT`  
$\alpha$ - broadband albedo for SW [TODO: avoid acronyms, only use them for at least 3 times in the document and properly introduce at first use] -only proxy $[-]$, source_name: `cfg:albedo`

**Suggested ranges (priors)** 
[TODO: put ranges directly after parameter description (previous section) not into separate ranges section. Make sure that value description is 
 understandable within the context of the parameter. E.g. here you describe parameter as "Priestley–Taylor coefficient" which has zero information what the parameter express, so 
 I have no clue how it is ralated to wet surfaces (in value range description); review whole document for this]
- $\alpha_{PT}$: $[1.0,\,1.6]$ (often near 1.26 over wet surfaces; increases in
  advective/dry conditions).  
- $\alpha$: $[0.05,\,0.35]$ (snow/bright soil higher; dark wet soil lower).

### Derived quantities
$T_c = T - T_0$ - air temperature in °C $[^\circ C]$  
$S_{\downarrow} = \Delta S^{acc}_{\downarrow}/\Delta t$ - SW flux $[W\,m^{-2}]$  
$R_n \approx (1-\alpha)\,S_{\downarrow}$ - net-radiation proxy $[W\,m^{-2}]$  
$e_s(T_c)$ - saturation vapor pressure $[kPa]$  
[TODO: formula here]
$\Delta$ [TODO: conflict with operator, use different symbol] - slope of sat. vapor pressure curve $[kPa\,^\circ C^{-1}]$
[TODO: formula here + explain what is saturation vaour pressure curve
### Model

$$
T_c = T - T_0 \quad
S_{\downarrow}=\frac{\Delta S^{acc}_{\downarrow}}{\Delta t} \quad
R_n \approx (1-\alpha)\,S_{\downarrow} \\
[TODO: remove redundant expressions]
e_s(T_c)=e_0\,\exp\!\left(\frac{a\,T_c}{T_c+b}\right) \quad
\Delta=\frac{c\,e_s(T_c)}{(T_c+b)^2} \quad
[TODO: Put these under the quantity definitions above]
\gamma \approx \gamma_0 \\
\lambda E=\alpha_{PT}\,\frac{\Delta}{\Delta+\gamma}\,R_n \quad
[TODO: \lambda looks like own symbol, use different notation like E_{ET} - "total evaporation heat flux"]
ET_p=\frac{\lambda E}{\lambda_v}
$$
[TODO: Put ET_p formula first, and then nominator subformula]
Explain the model $E_{ET}$ model, I have no idea what is the maning of the fraction with \Delta

---

## Net radiation

### Model overview
Improves the energy term by constructing a better [TODO: explain specificaly what effects it adds] net radiation $R_n$ using SW↓ 
[TODO: as with accronyms, symbols must be explained idealy on the first use, these symbosl are eithr not consistent or not described]
LW↓ and surface temperature. Output is improved $R_n$ for the PT core.  
Reference: [Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables (new)
$L^{acc}_{\downarrow}$ - downw. longwave, accumulated $[J\,m^{-2}]$,
source_name: `surface_thermal_radiation_downwards`  
$T_s$ - surface (skin) temperature $[K]$, source_name: `surface_temperature`

### New constants
$\sigma = 5.670374419\times10^{-8}$ - Stefan–Boltzmann $[W\,m^{-2}\,K^{-4}]$,
source_name: `cfg:sigma_SB`  

### New parameters
$\varepsilon$ - effective surface emissivity $[-]$, source_name: `cfg:emissivity`  
$\alpha$ - broadband albedo $[-]$, source_name: `cfg:albedo` (same as PT core)

**Suggested ranges (priors)**
- $\varepsilon$: $[0.92,\,0.99]$ (veg/wet soil high; dry mineral soil lower).  
- $\alpha$: $[0.05,\,0.35]$ (strongly surface-type and snow dependent).

### Derived quantities
$L_{\downarrow}=\Delta L^{acc}_{\downarrow}/\Delta t$ - LW↓ flux $[W\,m^{-2}]$  
$L_{\uparrow}=\varepsilon\,\sigma\,T_s^4$ - LW↑ flux $[W\,m^{-2}]$  
$S_{\uparrow}=\alpha\,S_{\downarrow}$ - SW↑ flux $[W\,m^{-2}]$

### Model
$$
L_{\downarrow}=\frac{\Delta L^{acc}_{\downarrow}}{\Delta t} \quad
L_{\uparrow}=\varepsilon\,\sigma\,T_s^4 \quad
S_{\uparrow}=\alpha\,S_{\downarrow} \\
R_n=(S_{\downarrow}-S_{\uparrow})+(L_{\downarrow}-L_{\uparrow})
$$

---

## Drying power

### Model overview
Computes atmospheric diagnostics (e.g. $VPD$, pressure-based $\gamma$) useful for
QA and for optional PT corrections. Outputs $VPD$, $\gamma(P)$, and $u_{10}$.  
Reference: [Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables (new)
$RH$ - 2 m relative humidity $[-]$, source_name: `relative_humidity_2m`  
$u_{10}$ - 10 m wind speed $[m\,s^{-1}]$, source_name: `wind_speed_10m`  
$P_{sl}$ - sea-level pressure $[Pa]$, source_name: `air_pressure_at_sea_level`

### New constants
$p_0 = 1000$ - Pa per kPa $[Pa\,kPa^{-1}]$, source_name: `cfg:Pa_per_kPa`

### New parameters
$k_\gamma$ - psychrometric proportionality $[kPa\,^\circ C^{-1}\,kPa^{-1}]$,
source_name: `cfg:k_gamma`  
$f(\cdot)$ - optional modifier for $\alpha_{PT}$ $[-]$,
source_name: `cfg:alphaPT_modifier`

**Suggested ranges (priors)**
- $k_\gamma$: narrow around FAO value $6.65\times10^{-4}$; allow $\pm 10\%$ if
  pressure is approximate (sea-level vs site).  
- $f(\cdot)$: keep near 1 unless you explicitly calibrate an advection term.

### Derived quantities
$P=P_{sl}/p_0$ - pressure in kPa $[kPa]$  
$e_a=RH\,e_s(T_c)$ - actual vapor pressure $[kPa]$  
$VPD=e_s(T_c)-e_a$ - vapor pressure deficit $[kPa]$

### Model
$$
P=\frac{P_{sl}}{p_0} \quad
\gamma=k_\gamma\,P \quad
e_a = RH\,e_s(T_c) \quad
VPD = e_s(T_c) - e_a \\
\alpha_{PT}^{eff}=f(u_{10},VPD)\,\alpha_{PT} \quad (\text{optional})
$$

---

## Partitioning

### Model overview
Provides a minimal split of potential ET into potential transpiration and soil
evaporation and a simple “excess water” flux used as a proxy for infiltration
input. Outputs $E_p$, $T_p$, $I_p$.  
Reference (dual-source / partitioning concepts):  
[Allen et al. (1998), FAO-56](https://www.fao.org/4/x0490e/x0490e00.htm)

### Input variables (new)
$P^{acc}$ - total precip, accumulated $[kg\,m^{-2}]$,
source_name: `precipitation_amount_accum`  
$P_r^{acc}$ - rainfall, accumulated $[kg\,m^{-2}]$,
source_name: `rainfall_amount_accum`  
$P_s^{acc}$ - snowfall, accumulated $[kg\,m^{-2}]$,
source_name: `snowfall_amount_accum`  
$SWE^{obs}$ - snow water equivalent (diag) $[kg\,m^{-2}]$,
source_name: `snow_water_equivalent`

### New constants
$\rho_w = 1$ - water-equivalent conversion $[mm\,/(kg\,m^{-2})]$,
source_name: `cfg:rho_w_mm_per_kgm2`

### New parameters
$f_T$ - transpiration fraction (potential) $[-]$, source_name: `cfg:f_T`

**Suggested ranges (priors)**
- $f_T$: $[0.1,\,0.9]$ (low in bare soil / dormant season; high in dense canopy).

### Derived quantities
$P=\rho_w\,\Delta P^{acc}/\Delta t$ - precip rate $[mm\,s^{-1}]$

### Model
$$
P=\rho_w\,\frac{\Delta P^{acc}}{\Delta t} \quad
T_p=f_T\,ET_p \quad
E_p=(1-f_T)\,ET_p \\
E_p \leftarrow \min(E_p,\,P) \quad
I_p=\max(P-E_p,\,0)
$$

---

# Further extensions (lumped ODE submodels)

All submodels below introduce **state variables** and produce modified fluxes
for Richards upper boundary and root uptake. Each section lists new inputs only.

---

## Snowpack ODE (storage + melt release)

### Model overview
Lumped snow store that delays water input until melt; outputs liquid input
$P_\ell$ for the soil surface.  
Reference (degree-day / SNOW-17 lineage):  
[Anderson (1973)](https://repository.library.noaa.gov/view/noaa/13507/noaa_13507_DS1.pdf)

### Input variables (new)
$P_r^{acc}$ - rainfall, accumulated $[kg\,m^{-2}]$,
source_name: `rainfall_amount_accum`  
$P_s^{acc}$ - snowfall, accumulated $[kg\,m^{-2}]$,
source_name: `snowfall_amount_accum`

### New constants
$\rho_w = 1$ - water-equivalent conversion $[mm\,/(kg\,m^{-2})]$,
source_name: `cfg:rho_w_mm_per_kgm2`  
$T_0 = 273.15$ - melt threshold offset $[K]$, source_name: `cfg:T0_K`

### New parameters
$c_m$ - melt factor $[mm\,s^{-1}\,K^{-1}]$, source_name: `cfg:melt_factor`  

**Suggested ranges (priors)**
- $c_m$: $[10^{-8},\,10^{-5}]$ (roughly $0.1$–$10$ mm/day/K; higher with strong
  radiation/advection, lower in shaded/calm conditions).

### State variables
$S(t)$ - snow water equivalent store on ground $[mm]$

### Derived quantities
$P_r=\rho_w\,\Delta P_r^{acc}/\Delta t$ - rain rate $[mm\,s^{-1}]$  
$P_s=\rho_w\,\Delta P_s^{acc}/\Delta t$ - snow rate $[mm\,s^{-1}]$

### Model
$$
P_r=\rho_w\,\frac{\Delta P_r^{acc}}{\Delta t} \quad
P_s=\rho_w\,\frac{\Delta P_s^{acc}}{\Delta t} \quad
m=c_m\,\max(T-T_0,0) \\
m \leftarrow \min\!\left(m,\frac{S}{\Delta t}+P_s\right) \quad
\frac{dS}{dt}=P_s-m \quad
P_\ell=P_r+m
$$

---

## Frozen-soil gate ODE (effective permeability)

### Model overview
Lumped freeze/thaw index that reduces effective liquid input to the soil when
the surface is frozen. Outputs $P_{\ell,eff}$.  
Reference (frozen-soil infiltration classes):  
[Gray et al. (1985)](https://research-groups.usask.ca/hydrology/documents/pubs/papers/gray_et_al_1985_3.pdf)

### Input variables (new)
$T_s$ - surface (skin) temperature $[K]$, source_name: `surface_temperature`

### New parameters
$\Delta T_f$ - freeze transition width $[K]$, source_name: `cfg:freeze_width_K`  
$\tau_f$ - freezing time scale $[s]$, source_name: `cfg:tau_freeze_s`  
$\tau_t$ - thawing time scale $[s]$, source_name: `cfg:tau_thaw_s`  
$p$ - sharpness exponent $[-]$, source_name: `cfg:freeze_exponent`

**Suggested ranges (priors)**
- $\Delta T_f$: $[0.5,\,5]$ K (controls how “binary” freezing appears).  
- $\tau_f,\tau_t$: $[10^4,\,10^6]$ s (hours to ~week; depends on soil heat
  capacity, snow insulation, radiation).  
- $p$: $[1,\,8]$ (higher makes infiltration collapse near $F\to 1$).

### State variables
$F(t)$ - frozen fraction / ice-blockage index $[-]$ (0–1)

### Derived quantities
$F^*(T_s)$ - equilibrium frozen fraction $[-]$  
$g_F$ - permeability multiplier $[-]$

### Model
$$
F^*(T_s)=\mathrm{clip}\!\left(\frac{T_0-T_s}{\Delta T_f},0,1\right) \quad
g_F=(1-F)^p \\
\frac{dF}{dt}=
\begin{cases}
\frac{F^*(T_s)-F}{\tau_f}, & F^*(T_s)>F \\
\frac{F^*(T_s)-F}{\tau_t}, & F^*(T_s)\le F
\end{cases}
\quad
P_{\ell,eff}=g_F\,P_\ell
$$

---

## Canopy interception ODE (rain/snow storage + drip)

### Model overview
Lumped canopy water and canopy snow stores that delay delivery to the ground and
allow wet-canopy losses. Outputs modified ground inputs $P_{\ell}$ and $P_s$.  
Reference (canopy storage model):  
[Rutter et al. (1971)](https://www.sciencedirect.com/science/article/pii/0002157171900343)

### Input variables (new)
(uses $P_r$, $P_s$, $T$ from earlier sections)

### New parameters
$C_r$ - canopy liquid storage capacity $[mm]$, source_name: `cfg:canopy_Cr_mm`  
$C_s$ - canopy snow storage capacity $[mm]$, source_name: `cfg:canopy_Cs_mm`  
$\tau_r$ - drip time scale $[s]$, source_name: `cfg:canopy_tau_r_s`  
$\tau_s$ - snow unloading time scale $[s]$, source_name: `cfg:canopy_tau_s_s`  
$c_{m,c}$ - canopy snow melt factor $[mm\,s^{-1}\,K^{-1}]$,
source_name: `cfg:canopy_melt_factor`  
$k_i$ - wet-canopy evap. rate constant $[s^{-1}]$,
source_name: `cfg:wet_canopy_evap_rate`

**Suggested ranges (priors)**
- $C_r$: $[0.1,\,2]$ mm (structure, LAI; conifers often higher than grass).  
- $C_s$: $[0.5,\,10]$ mm (snow loading; depends on canopy type).  
- $\tau_r$: $[10^3,\,10^5]$ s (minutes–day; wind and structure).  
- $\tau_s$: $[10^4,\,10^6]$ s (hours–week; snow cohesion and wind).  
- $c_{m,c}$: similar scale as $c_m$ (radiation/wind exposure).  
- $k_i$: $[10^{-7},\,10^{-4}]$ s$^{-1}$ (sets wet-canopy evaporation strength).

### State variables
$W_r(t)$ - canopy liquid water store $[mm]$  
$W_s(t)$ - canopy snow store $[mm]$

### Derived quantities
$I_r,TF_r$ - intercepted rain and throughfall $[mm\,s^{-1}]$  
$I_s,TF_s$ - intercepted snow and throughfall $[mm\,s^{-1}]$  
$D_r$ - drip $[mm\,s^{-1}]$  
$U_s$ - snow unloading $[mm\,s^{-1}]$  
$M_{cs}$ - canopy snow melt $[mm\,s^{-1}]$  
$E_i$ - wet-canopy evaporation loss $[mm\,s^{-1}]$

### Model
$$
I_r=\min\!\left(P_r,\max\!\left(0,\frac{C_r-W_r}{\Delta t}\right)\right) \quad
TF_r=P_r-I_r \quad
D_r=\frac{W_r}{\tau_r} \quad
E_i=k_i\,W_r \\
I_s=\min\!\left(P_s,\max\!\left(0,\frac{C_s-W_s}{\Delta t}\right)\right) \quad
TF_s=P_s-I_s \quad
U_s=\frac{W_s}{\tau_s} \quad
M_{cs}=c_{m,c}\,\max(T-T_0,0) \\
M_{cs}\leftarrow \min\!\left(M_{cs},\frac{W_s}{\Delta t}\right) \quad
\frac{dW_r}{dt}=I_r-E_i-D_r \quad
\frac{dW_s}{dt}=I_s-M_{cs}-U_s \\
P_\ell = TF_r + D_r + M_{cs} + m \quad
P_s = TF_s + U_s
$$

---

## Vegetation stress ODE (transpiration demand limiter)

### Model overview
Lumped root-zone “available water” index that reduces transpiration demand before
it is applied as Richards root uptake. Outputs stressed transpiration $T_a$.  
Reference (macroscopic root water uptake / stress concepts):  
[Skaggs et al. (2006)](https://www.pc-progress.com/Documents/RVGenugten/Skaggs_Root_water_uptake_AWM.pdf)

### Input variables (new)
(uses $ET_p$ from PT core; uses $P_{\ell,eff}$ from freeze/snow gating)

### New parameters
$\theta_w$ - wilting threshold index $[mm]$, source_name: `cfg:theta_w_mm`  
$\theta_{fc}$ - non-stress index (field-capacity proxy) $[mm]$,
source_name: `cfg:theta_fc_mm`  
$f_T$ - potential transpiration fraction $[-]$, source_name: `cfg:f_T`  
$a$ - fraction of surface input benefiting roots within a week $[-]$,
source_name: `cfg:root_recharge_fraction`  
$b$ - leak/drain time-scale rate $[s^{-1}]$, source_name: `cfg:root_leak_rate`

**Suggested ranges (priors)**
- $\theta_w,\theta_{fc}$: site/soil dependent; set from rooting depth and a
  plausible plant-available water capacity (order 10–200 mm).  
- $a$: $[0,\,1]$ (small if most events bypass roots; larger for shallow roots).  
- $b$: $[10^{-7},\,10^{-5}]$ s$^{-1}$ (week-scale memory; higher drains faster).

### State variables
$\theta_r(t)$ - root-zone available water index $[mm]$

### Derived quantities
$f_s(\theta_r)$ - stress factor $[-]$  
$T_p$ - potential transpiration $[mm\,s^{-1}]$  
$T_a$ - stressed transpiration $[mm\,s^{-1}]$

### Model
$$
f_s(\theta_r)=\mathrm{clip}\!\left(
\frac{\theta_r-\theta_w}{\theta_{fc}-\theta_w},0,1\right) \quad
T_p=f_T\,ET_p \quad
T_a=f_s(\theta_r)\,T_p \\
\frac{d\theta_r}{dt}=a\,P_{\ell,eff}-T_a-b\,\theta_r
$$

---

## Summary of new ODE states (all extensions)
- Snowpack: $S(t)$  
- Frozen gate: $F(t)$  
- Canopy: $W_r(t)$, $W_s(t)$  
- Vegetation stress: $\theta_r(t)$

These states produce modified fluxes for coupling:
- liquid input to soil: $P_{\ell,eff}$  
- transpiration sink demand: $T_a$
