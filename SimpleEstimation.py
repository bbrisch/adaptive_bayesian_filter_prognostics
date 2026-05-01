import src.utils as utils
import numpy as np
import matplotlib.pyplot as plt

# instance the particle filter and state function
N_particles = 100
state_eq = lambda x, p: x+p + np.random.normal(0,1, len(x))
param_eq = lambda p: p + np.random.normal(0, 1, len(p))

gt_param = 2
y_gt = 0.001
gt_eq1 = lambda y: y+gt_param
gt_eq2 = lambda y: y-gt_param



sigma_measurements = 1
# obs_params = [0,sigma_measurements]

PF = utils.SIR_PF(state_fn = state_eq,param_fn=param_eq, N_particles=N_particles, sigma_measurements=sigma_measurements)


# Start the simulation to estimate the parameter

gt = 1
alpha_0 =1.5

x_0 = np.random.uniform(gt*(1-alpha_0), gt*(1+alpha_0), N_particles)
p_0 = np.random.uniform(0,1.5, N_particles)
PF.init_state(x_0, p_0) 
# print(PF.w.sum())

mean_x = []
q_l_x = []
q_u_x = []

mean_p = []
q_l_p = []
q_u_p = []

gt_list = []

for i in range(1000):
    
    if i<500:
        y_gt = gt_eq1(y_gt)
    else:
        y_gt = gt_eq2(y_gt)
    
    gt_list.append(y_gt)
    y = np.random.normal(y_gt, 1, 1)
    PF.update(y)
    # print(PF.w.sum())
    # if i%100 == 0:
    #     mean = sum(PF.x*PF.w)
    #     var = sum((PF.x-mean)**2*PF.w)
    #     print(f'expected: {mean}\tvar: {var}')
    #     plt.figure()
    #     for x in PF.x:
    #         plt.axvline(x, color = 'b')
    #     plt.axvline(y, color = 'r')
    #     plt.show()

    mean_x.append(PF.mean()[0])
    
    q_l_x.append(PF.quantile(0.05)[0])
    q_u_x.append(PF.quantile(0.95)[0])
    
    mean_p.append(PF.mean()[1])
    q_l_p.append(PF.quantile(0.05)[1])
    q_u_p.append(PF.quantile(0.95)[1])
    
    

plt.figure()
plt.plot(mean_x, color = 'b')

plt.plot(q_l_x, color = 'r')
plt.plot(q_u_x, color = 'r')

plt.plot(gt_list)
plt.ylabel('x')
plt.show()


plt.figure()
plt.plot(mean_p, color = 'b')

plt.plot(q_l_p, color = 'r')
plt.plot(q_u_p, color = 'r')

plt.ylabel('p')
plt.axhline(gt_param)
plt.show()

    

