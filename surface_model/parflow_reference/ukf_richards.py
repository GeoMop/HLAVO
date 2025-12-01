import pde  # py-pde
import numpy as np
import matplotlib.pyplot as plt

import dataclasses
from notebooks.Richards.richards_sim import mu_from_h, h_from_mu, RichardsPDE, Hydraulic
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints

# mes_1_loc = 0.1
# mes_2_loc = 0.3

mes_locations = [0.1, 0.3]
#mes_locations = [0.1]

def get_field_measurements(field, locations):
    measurements = []
    for loc in locations:
        measurements.append(field.data[loc])
    return measurements


def get_measurements(storage, time_steps, space_step=1):
    measurements = []
    for t_step in time_steps:
        print('t step ', t_step)
        field = storage._get_field(t_index=t_step)

        loc_measurements = get_field_measurements(field, [int(mes_loc/space_step) for mes_loc in mes_locations])
        #loc_measurements = get_field_measurements(field, [int(mes_1_loc/space_step), int(mes_2_loc/space_step)])
        measurements.append(loc_measurements)
        #measurements.append((field.data[int(mes_1_loc/space_step)], field.data[int(mes_2_loc/space_step)]))
    print("measurements ", measurements)
    return measurements
    #print("field ", field.data[int(mes_1_loc/space_step)])


def add_noise(measurements, level=0.05):
    noisy_measurements = np.zeros(measurements.shape)
    for i in range(measurements.shape[1]):
        for idx, mes_val in enumerate(measurements[:, i]):
            noise = np.random.uniform(-level* mes_val, level*mes_val)
            noisy_measurements[idx, i] = measurements[idx, i] + noise
    return noisy_measurements


sand = Hydraulic(
    th_s=0.38,
    th_0=0.05,
    k_s=77.7 * 0.01 / 24 / 3600,  # [cm/d] -> [m/s]
    alpha=0.035 * 100,  # alpha [cm^{-1}] -> lmbd [m^{-1}]
    n=1.6167)

# sand: 10%, silt 80%, clay 10%
silt = Hydraulic(
    th_s=0.47,
    th_0=0.06,
    k_s=30.8 * 0.05 / 24 / 3600, #30.8 * 0.01 / 24 / 3600,  # [cm/d] -> [m/s],   0,000017824, 0,000003565

    alpha=0.0058 * 100,  # alpha [cm^{-1}] -> lmbd [m^{-1}]
    n=1.6745)

model = silt


#############################
### Generate measurements ###
#############################
L = 1
z_step = 0.05
grid = pde.CartesianGrid([[0, L]], int(L / z_step))

h = pde.ScalarField.from_expression(grid, "-10*(1-x) + -2*x")
print("h for initital state ", h)
state = mu_from_h(h)

eq_raining = RichardsPDE(
        hydraulic=model,
        flux_top=2.5 * 1e-3,   # 0.5 mm/h
        h_bot=-2 # [m]
        )

eq_not_raining = RichardsPDE(
        hydraulic=model,
        flux_top=0,
        h_bot=-2 # [m]
        )

n_days = 10  # 5 days
hours_of_rain_per_day = 4
#hours_without_rain_per_day = 32



rain_no_rain_time_ranges = {(0, 4): "rain", (4, 32): "no_rain", (32,36): "rain",  (36,64): "no_rain", (64,68): "rain",
               (68,96): "no_rain", (96,100): "rain", (100,128): "no_rain", (128, 132): "rain", (132, 160): "no_rain",}

rain_no_rain_time_ranges = {(0, 6): "rain", (6, 30): "no_rain", (30,36): "rain",  (36,60): "no_rain", (60,66): "rain",
               (66,90): "no_rain", (90,96): "rain", (96,120): "no_rain", (120, 126): "rain", (126, 150): "no_rain",}

rain_no_rain_time_ranges = {(0, 2): "rain", (2, 26): "no_rain", (26,30): "rain",  (30,54): "no_rain", (54,60): "rain",
               (60,84): "no_rain", (84,92): "rain", (92,116): "no_rain", (116, 126): "rain", (126, 150): "no_rain",}

rain_no_rain_time_ranges = {(0, 8): "rain", (8, 44): "no_rain", (44,52): "rain",  (52,88): "no_rain", (88,96): "rain",
               (96,132): "no_rain", (132,140): "rain", (140,176): "no_rain"}

# rain_no_rain_time_ranges = {(0, 2): "rain", (2, 32): "no_rain", (32,34): "rain",  (34,64): "no_rain", (64,66): "rain",
#                (66,96): "no_rain", (96,98): "rain", (98,128): "no_rain", (128, 130): "rain", (130, 160): "no_rain",}


storage = pde.MemoryStorage(write_mode="append")


if rain_no_rain_time_ranges is not None:
    print("rain_no_rain_time_ranges")
    for time_range, status in rain_no_rain_time_ranges.items():
        if status == "rain":
            # Raining period
            state = eq_raining.solve(state, t_range=time_range, dt=0.05, method='scipy',
                                     tracker=["progress", storage.tracker(interval=0.1)])

            print("raining time: ({}, {})".format(time_range[0], time_range[1]))

            # storage.append(state)
        else:
            # Not raining period
            state = eq_not_raining.solve(state, t_range=time_range, dt=0.05,
                                         method='scipy',
                                         tracker=["progress", storage.tracker(interval=0.1)])

            print("not raining time: ({}, {})".format(time_range[0], time_range[1]))

else:
    for i in range(0, n_days):
        # Raining period
        state = eq_raining.solve(state, t_range=(i*24, i*24 + hours_of_rain_per_day), dt=0.05, method='scipy',
                          tracker=["progress", storage.tracker(interval=0.1)])

        #print("raining time: ({}, {})".format(i * 24, i*24 + hours_of_rain_per_day))

        #storage.append(state)

        # Not raining period
        state = eq_not_raining.solve(state, t_range=(i*24 + hours_of_rain_per_day, (i+1)*24), dt=0.05, method='scipy',
                                  tracker=["progress", storage.tracker(interval=0.1)])

        print("not raining time: ({}, {})".format(i*24 + hours_of_rain_per_day, (i+1)*24))

        #print("state after not raining ", state)

        #storage.append(state)

        #print("storage.data_shape ", storage.data_shape)

#print("storage.data_shape ", storage._get_field(0))
# plot pressure head evolution


storage = storage.apply(h_from_mu)

#storage.data

measurements = get_measurements(storage, time_steps=range(0, 24*7, 1), space_step=z_step)
noisy_measurements = add_noise(np.array(measurements), level=0.05)

measurements = np.array(measurements)
noisy_measurements = np.array(noisy_measurements)

times = np.arange(1, len(measurements) + 1, 1)


plt.scatter(times, measurements[:, 0], marker="o", label="measurements")
plt.scatter(times, noisy_measurements[:, 0], marker='x',  label="noisy measurements")
plt.xlabel("time")
plt.ylabel("pressure head")
plt.legend()
plt.show()

if measurements.shape[1] > 1:
    plt.scatter(times, measurements[:, 1], marker="o", label="measurements")
    plt.scatter(times, noisy_measurements[:, 1], marker='x',  label="noisy measurements")
    plt.xlabel("time")
    plt.ylabel("pressure head")
    plt.legend()
    plt.show()


residuals = noisy_measurements - measurements
print("residuals ", residuals)
#residuals = residuals * 2

measurement_noise_covariance = np.cov(residuals, rowvar=False)
print("measurement noise covariance ", measurement_noise_covariance)
#print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
#exit()
pde.plot_kymograph(storage)



#####################
### Kalman filter ###
#####################
h = pde.ScalarField.from_expression(grid, "-10*(1-x) + -2*x")
initial_state = mu_from_h(h)


n_state_function_calls = 0


def state_transition_function(field_data, dt, rain=True):
    #print("state function call")
    global n_state_function_calls
    n_state_function_calls += 1
    #print("state transition_function: state ", field_data)
    #print("dt ", dt)
    #print("rain ", rain)

    state = pde.ScalarField(grid, field_data)

    if rain:
        state = eq_raining.solve(state, t_range=1, dt=0.05, method='scipy',
                                 tracker=["progress", storage.tracker(interval=0.1)])
    else:
        # Not raining period
        state = eq_not_raining.solve(state, t_range=1, dt=0.05, method='scipy',
                                     tracker=["progress", storage.tracker(interval=0.1)])

    #print("not raining time: ({}, {})".format(i * 24 + hours_of_rain_per_day, (i + 1) * 24))

    return state.data
    #return state


def measurement_function(state):
    measurements = get_field_measurements(state,  [int(mes_loc/z_step) for mes_loc in mes_locations])
    print("obtained measurements ", measurements)
    return measurements


num_locations = grid.shape[0]
n = num_locations
dt = 60*60  # Time between steps in seconds
dim_z = len(mes_locations)  # Number of measurement inputs
initial_covariance = np.eye(n) * 1e4
#initial_covariance = np.cov(noisy_measurements, rowvar=False)

print("num locations ", num_locations)

#sigmas = JulierSigmaPoints(n=n, kappa=1)
sigma_points = MerweScaledSigmaPoints(n=num_locations, alpha=1e-3, beta=2.0, kappa=3-len(mes_locations))
# Initialize the UKF filter
ukf = UnscentedKalmanFilter(dim_x=n, dim_z=dim_z, dt=dt, fx=state_transition_function, hx=measurement_function, points=sigma_points)
ukf.Q = np.ones(num_locations) * 5e-6
#ukf.Q = Q_discrete_white_noise(dim=1, dt=1.0, var=1e-6, block_size=num_locations)  # Process noise covariance
ukf.R = measurement_noise_covariance

# best setting so far initial_covariance = np.eye(n) * 1e4, ukf.Q = np.ones(num_locations) * 1e-7


# print("state ", state)
# print("state.data[int(0.3/0.5)]" , state.data[int(0.3/0.05)])
# print("state data " , state.data)
#field.data[int(mes_2_loc/space_step)]

print("initial_state.data ", initial_state.data)

ukf.x = initial_state.data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
ukf.P = initial_covariance  # Initial state covariance matrix

pred_loc_measurements = []
raining = True
i = 0
i_rain = 0
# Assuming you have a measurement at each time step
for measurement in noisy_measurements:
    if i == 24:
        i_rain = 0
        i = 0
    if i_rain > 3:
        ukf.predict(rain=False)
        print("no rain, i: {}".format(i))
    else:
        ukf.predict(rain=True)
        i_rain += 1
        print("rain, i: {}".format(i))
    i += 1

    print("measurement ", measurement)
    ukf.predict(rain=True)
    ukf.update(measurement)
    print("Estimated State:", ukf.x)
    est_loc_measurements = get_field_measurements(ukf.x,  [int(mes_loc/z_step) for mes_loc in mes_locations])
    print("est loc measurements ", est_loc_measurements)

    pred_loc_measurements.append(est_loc_measurements)
    #zs.append(measurement)



pred_loc_measurements = np.array(pred_loc_measurements)
noisy_measurements = np.array(noisy_measurements)

print("state_loc_measurements ", pred_loc_measurements)
print("noisy_measurements ", noisy_measurements)

times = np.arange(1, pred_loc_measurements.shape[0] + 1, 1)

print("times.shape ", times.shape)
print("xs[:, 0].shape ", pred_loc_measurements[:, 0].shape)


plt.scatter(times, pred_loc_measurements[:, 0], marker="o", label="predictions")
plt.scatter(times, measurements[:, 0], marker='x',  label="measurements")
plt.scatter(times, noisy_measurements[:, 0], marker='x',  label="noisy measurements")
plt.legend()
plt.show()

print("Var noisy measurements ", np.var(pred_loc_measurements[:, 0]))
print("Var predictions ", np.var(pred_loc_measurements[:, 0]))

if measurements.shape[1] > 1:
    plt.scatter(times, pred_loc_measurements[:, 1], marker="o", label="predictions")
    plt.scatter(times, measurements[:, 1], marker='x',  label="measurements")
    plt.scatter(times, noisy_measurements[:, 1], marker='x',  label="noisy measurements")
    plt.legend()
    plt.show()


print("MSE predictions vs measurements loc 0 ", np.mean((measurements[:, 0] - pred_loc_measurements[:, 0])**2))
print("MSE noisy measurements vs measurements loc 0", np.mean((measurements[:, 0] - noisy_measurements[:, 0])**2))

print("MSE predictions vs measurements loc 1 ", np.mean((measurements[:, 1] - pred_loc_measurements[:, 1])**2))
print("MSE noisy measurements vs measurements loc 1", np.mean((measurements[:, 1] - noisy_measurements[:, 1])**2))

#print("MSE predictions vs measurements ", np.mean((measurements - pred_loc_measurements)**2))
#print("MSE noisy measurements vs measurements ", np.mean((measurements - noisy_measurements)**2))

print("Var noisy measurements ", np.var(pred_loc_measurements[:, 1]))
print("Var predictions ", np.var(noisy_measurements[:, 1]))


print("N state function calls ", n_state_function_calls)
