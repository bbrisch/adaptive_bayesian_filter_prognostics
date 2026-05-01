
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from src.utils import Dataloader,SIR_PF

np.random.seed(42)

# Load data
dataloader = Dataloader('data_NE', 'percentual', thresh=0.2)

# Fit obs_eq
def poli_fn(hi, a,b,c):
    return a*hi**3+b*hi**2+c*hi

f_train = np.random.choice(list(dataloader.ids), 7, replace=False )
f_test = [i  for i in dataloader.ids if not i in f_train]




data_train = []
hi_train = []
for f in f_train:
    data = dataloader.load_id(f)
    data_train.extend(data)
    hi_train.extend(np.linspace(0,1, len(data)))
    
msg= curve_fit(poli_fn, hi_train,data_train,p0=(1,1,1), full_output=False)
popt = msg[0]
print(msg)

print('\n++++++++++++++++++++++++++\n')

# get noise level and damage accumulation rates
errors = []
damage_rate = []
for f in f_train:
    data = dataloader.load_id(f)
    h = np.linspace(0,1,len(data))
    y = poli_fn(h, *popt)
    errors.extend(data-y)
    damage_rate.append(1/len(data))
t = np.linspace(0,1,100)
print(damage_rate)

plt.figure()
plt.plot(t, poli_fn(t, *popt), color = 'b')
for f in f_train:
    data = dataloader.load_id(f)
    h = np.linspace(0,1,len(data))
    plt.plot(h,data, color = 'grey', alpha = 0.5)
plt.ylabel('y')
plt.xlabel('h')
plt.show()
    
print('error mean and std:\t', np.mean(errors), np.std(errors))
print('damage rate mean and std:\t', np.mean(damage_rate), np.std(damage_rate))


# Setup the update functinos for the PF 
N_particles = 500
obs_std = 1e-5
# obs_std = np.std(errors)

w_param = 1e-8
print('AAAA',obs_std)

obs_eq = lambda x: poli_fn(x, *popt)
# state_eq = lambda x, p:np.clip( x+p + np.random.normal(0,1e-12, N_particles), 0, 1)
state_eq = lambda x, p:np.clip( x+p, 0, 1)
param_eq = lambda p: np.clip(p + np.random.normal(0, w_param, N_particles),0, np.inf)




# Setup the priors
x_0 = np.random.uniform(0, 0.1, N_particles) # We should be confident that it starts at 0 -> No past uncertainty
# p_0 = np.random.choice(damage_rate, N_particles)
p_0 = np.clip(gaussian_kde(damage_rate ).resample(N_particles),0, np.inf).ravel()


# Instance de PF
particle_filter = SIR_PF(obs_eq, state_eq, param_eq, N_particles, obs_std)
particle_filter.init_state(x_0, p_0)



# Start the filtering process
print('\n++++++++++++++++++++++++++\n')

data_test = dataloader.load_id(f_test[0])
p_gt = 1/len(data_test)
thresh = 1-1e-3


mean_x = []
q_l_x = []
q_u_x = []

mean_y = []
q_l_y = []
q_u_y = []

mean_p = []
q_l_p = []
q_u_p = []


for i,y in enumerate(data_test):
    particle_filter.update(y, thresh)
    
    mean_x.append(particle_filter.mean()[0])
    q_l_x.append(particle_filter.quantile(0.05)[0])
    q_u_x.append(particle_filter.quantile(0.95)[0])
    
    mean_p.append(particle_filter.mean()[1])
    q_l_p.append(particle_filter.quantile(0.05)[1])
    q_u_p.append(particle_filter.quantile(0.95)[1])
    
    mean_y.append(particle_filter.mean()[2])
    q_l_y.append(particle_filter.quantile(0.05)[2])
    q_u_y.append(particle_filter.quantile(0.95)[2])
    
    
    # if i%10 == 0:
    #     w_param = w_param*0.1
    #     print(w_param)
    #     param_eq = lambda p: np.clip(p + np.random.normal(0, w_param, N_particles),0, np.inf)

        

        
    #     prog = particle_filter.prognostics()
    #     t = np.arange(len(data_test))
        
    #     plt.figure()
    #     plt.plot(t, prog.evaluate(t))
    #     plt.tight_layout()
    #     plt.show()
    
    # print(particle_filter.mean())



t = np.arange(len(data_test))

plt.figure()
plt.plot(t, mean_y, color = 'b')
plt.fill_between(t, q_l_y, q_u_y, color = 'b', alpha =0.2)
plt.plot(data_test, color = 'r')
plt.ylabel('y')
plt.show(block=False)


plt.figure()
plt.plot(t, mean_x, color = 'b')
plt.fill_between(t, q_l_x, q_u_x, color = 'b', alpha =0.2)
plt.plot(np.linspace(0,1, len(data_test)), color = 'r')
plt.ylabel('x')
plt.show(block=False)

plt.figure()
plt.plot(t, mean_p, color = 'b')
plt.fill_between(t, q_l_p, q_u_p, color = 'b', alpha =0.2)
plt.axhline(p_gt, color = 'r')
plt.ylabel('p')
plt.ylim(0, max(damage_rate)*2)

plt.show()
