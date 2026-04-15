import src
import numpy as np
import matplotlib.pyplot as plt


# instance the particle filter and state function
N_particles = 10000
state_eq = lambda x: x
sigma_measurements = 0.1
obs_params = [0,sigma_measurements]

PF = src.SIR_PF(state_fn = state_eq, N_particles=N_particles, obs_params=obs_params)


# Start the simulation to estimate the parameter

parameter = 1
alpha_0 = 3

x_0 = np.random.uniform(parameter*(1-alpha_0), parameter*(1+alpha_0), N_particles)+0.75
PF.init_state(x_0) 
# print(PF.w.sum())

mean_x = []
q_l = []
q_u = []

for i in range(1000):
    
    y = np.random.normal(parameter, sigma_measurements, 1)
    PF.update(y)
    # print(PF.w.sum())
    mean_x.append(PF.mean())
    q_l.append(PF.quantile(0.05))
    q_u.append(PF.quantile(0.95))
    
    

plt.figure()
plt.plot(mean_x, color = 'b')

plt.plot(q_l, color = 'r')
plt.plot(q_u, color = 'r')

plt.axhline(parameter)
plt.show()

    

