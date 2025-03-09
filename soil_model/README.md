# Unscanted Kalman Filter applied to Richards equation

Goal is to predict deep infiltration.
Test problem: 1D 5m depth, Dirichlet on bottom, variable flux on the top (e.g. random noise > 0.5)
Measurement: 5 moisture sensors at 0.1, 0.2, 0.4, 0.6, 1.0 m depth.
Test variables: 
- moisture at 0.5, 0.8, 2.0, 4.0 m depth.
- flux at 1m, 2m, 5m 

Goal is to calibrate all vG parameters + linear sensor coeficients
1. with known flux on top
2. with unknown flux on top


State variables:
- pressure head
- moisture values
- van Genuchten parameters (assuming constant over whole domain)
Resource[https://www.researchgate.net/figure/The-van-Genuchten-Mualem-model-parameters-for-simulated-soils_tbl1_329050595]- 
taking all sandy soil variants not complete sand
Initial state, independent:

theta_r=N[0.05, 0.9] @ 95
theta_s=N[0.35, 0.45] @ 95
n=N[1.5, 2.3] @ 95
alpha=N[0.02, 0.12] @ 95    [1/cm]
Ks=LN[0.01, 0.3] @ 95      [cm/min

Parameters for the synthetic measurements:
- van Gnuchten parameters for sandy loam:
- theta_r=0.065
- theta_s=0.41
- alpha=0.75 1/cm
- n=1.89
- Ks=0.0737 cm/min

změna na 100cm cca 400 minut
časový krok 1/12 min.

TODO:
Goals:
- 

- arbitraty timestepping in both Kalman and Richards models
  case: realistic conductivity and other parameters, still short period

- able to combine pressure, saturation, velocity train/test measurements
  case: 1D, 5m, 4 moisture sensors, 2 moisture 3 pressure 1 velocity tests

- move measurement cache to abstract problem, add all model inputs as
  pressure profile, weak bc conditions, vG parameters (constant in space for near future) 

- we want to build a database to form a surrogate model


- count number of model evaluations, monitor total model time, 
  get time of whole calculation


- parallelization, own model implementation
- real weather data, top BC (deterministic)
- 150 day UKF applied to synthetic measurement with real waether data, 
  full set of vG and measurement parameters; very low process and measurement noise 
- same for laboratory data one month

 
- merge soil model to surface model
- plot P matrix in more time steps
- fix the plot it shows nearly independent values of the pressure vector, but 
  they should be rather correlated
- alternative P plot for pressure profile, eigenvec decomposition:
  plot eigen values + profiles of eigne vectors for the first 10 components
- model is overconfident possibly due to very small Q
- Ks and rain are a bit unrealistic, but that is porblem of time scale, we need
  a way to set timestepping independently

- MS: remove measurements from state, save them in ToyProblem to a dict
  with state as a key and measurement as a value
  - MS: add support for test velocity
  ===
  Process noise:
  pressure: C dh/dt = model + err
  integrating h(t+dt) = h(t) + dt/C*model + dt/C * err
  E (H - h_m)^2
- allow arbitrary rain (flux) input
- Fix time unit to hours. 

- "synthetic_data_params" 
  updates the params dict, nonfixed will be fixed to 
  mean
- Plots for: test values, model parameters vs. prior and syntetic model data
  (seems partly done)
- synthetic rain procedure

Then we have to play with, R, Q, Kalman time step, and length of simulation
- How to get cov matrices for model and measurements (Q, R)?
- Adding a process noise to the synthetic data, assuming 
  a smooth gaussian external flux (roots, lateral flow, soil inhomogeneity).
  White noise in time.
  Could try to incorporate some of those into stat in future.

Questions:
- How Kalman time step affects convergence, how to choose it?
  How the simulation time compares to the Kalman processing time?
  Simpler Q choice is justifable for small time steps.

Future:
Acceleration:
- Own implementation.
- Use a dynamicaly build surogate model to predict sigmapoints
  with low weight. The simplest lienar model would be enough for
  the components that has contribution below the process noise.
  (has to be precised - we must project Q matrix to the eigen vectors to
   get the process noise)
- Process noise:
  - General approach:
    Input state -> result state
    with noise introduced to specific place of the model,
    i.e. parameters not modeled
    Q is nonlinear function of input state, but we 
    use an awerage in the KAlman just to have idea about actual 
    noise level of the model. 
  - We can bulild a hierarchy of models of different complexity
    can use precise models for most important components and 
    low fidelity models for less important components.
  - It could be viewed as having small number of components
    for constructing P  with precise model, but having nonlineear
    Q[x] predicted by the low fidelity model.
