import os
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from src.utils import Dataloader,SIR_PF, sample_crps, expected_rul

np.random.seed(42)

# Load data
dataloader = Dataloader('data_NE', 'stiffness_red', thresh=0.15)


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

print('\n++++++++++++++++++++++++++\n')

# get noise level and damage accumulation rates
errors = []
damage_rate = []
lengths = []
for f in f_train:
    data = dataloader.load_id(f)
    lengths.append(len(data))
    h = np.linspace(0,1,len(data))
    y = poli_fn(h, *popt)
    errors.extend(data-y)
    damage_rate.append(1/len(data))
t = np.linspace(0,1,100)
# print(damage_rate)



# Start the filtering process
print('\n++++++++++++++++++++++++++\n')

data_test = dataloader.load_id(f_test[0])
prognostics = True
prog_freq = 10
N_particles = 200

M = 50

w_param = 1e-5
obs_std = np.std(errors)
obs_eq = lambda x: poli_fn(x, *popt)

adaptive = True
if adaptive:
    thresh = 0.5
    smoothing = True
    
    root_folder = os.path.join('results', 'adaptive')
    
    state_eq = lambda x, p:np.clip( x+p + np.random.normal(0,1e-5, N_particles), 0, 1)
    param_eq = lambda p: np.clip(p + np.random.normal(0, w_param, N_particles),0, np.inf)
    backprop_eq = lambda x,p,order: np.clip(x-p*order,0,1)


else:
    thresh = 0.9
    smoothing = False
    state_eq = lambda x, p:np.clip( x+p + np.random.normal(0,1e-5, N_particles), 0, 1)
    param_eq = lambda p: p
    backprop_eq = lambda x,p,order:None
    root_folder = os.path.join('results', 'normal')

os.makedirs(root_folder, exist_ok=True)






# Setup the priors
# x_0 = np.random.uniform(0, 0.1, N_particles) # We should be confident that it starts at 0 -> No past uncertainty
x_0 = np.zeros(N_particles)
p_0 = np.random.choice(damage_rate, N_particles)
# p_0 = np.clip(gaussian_kde(damage_rate ).resample(N_particles),0, np.inf).ravel()

# Instance de PF
particle_filter = SIR_PF(obs_eq, 
                         state_eq, 
                         param_eq,
                         backprop_eq, 
                         N_particles, 
                         obs_std, 
                         thresh, 
                         M
                         )
particle_filter.init_state(x_0, p_0)





rul = len(data_test)- np.arange(len(data_test))
p_gt = 1/len(data_test)

rul_pred = []
rul_pred_ind = []

mae_list = []
crps_list = []

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
    
    
    
    particle_filter.update(y)
    particle_filter.y_buffer.append(y)
    
    
    
    if i%M == 0 and i!=0 and smoothing:        
        particle_filter.chi2_smoothing()
    
    if prognostics and i%prog_freq == 0:
        print(i, len(data_test))
        ttf, w = particle_filter.prognostic()
        
        expected = expected_rul(ttf,w)
        
        rul_pred.append(expected)
        rul_pred_ind.append(i)
        mae_list.append(abs(rul[i]-expected))
        crps_list.append(sample_crps(y,ttf, w))
        
        
        

    
    mean_x.append(particle_filter.mean()[0])
    q_l_x.append(particle_filter.quantile(0.05)[0])
    q_u_x.append(particle_filter.quantile(0.95)[0])
    
    mean_p.append(particle_filter.mean()[1])
    q_l_p.append(particle_filter.quantile(0.05)[1])
    q_u_p.append(particle_filter.quantile(0.95)[1])
    
    mean_y.append(particle_filter.mean()[2])
    q_l_y.append(particle_filter.quantile(0.05)[2])
    q_u_y.append(particle_filter.quantile(0.95)[2])
    
    

    
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




plt.figure()
plt.title(f'mae: {np.mean(mae_list) :.3f}, crps {np.mean(crps_list):.3f}' )
plt.plot(rul_pred_ind,rul_pred, color = 'b')
plt.plot(rul, color = 'r')
plt.ylabel('Rul')
plt.savefig(os.path.join(root_folder, 'prognostics.png'))
plt.show(block=False)


t = np.arange(len(data_test))
plt.figure()
plt.plot(t, mean_y, color = 'b')
plt.fill_between(t, q_l_y, q_u_y, color = 'b', alpha =0.2)
plt.plot(data_test, color = 'r')
plt.ylabel('y')
plt.savefig(os.path.join(root_folder, 'obs_estim.png'))
plt.show(block=False)


plt.figure()
plt.plot(t, mean_x, color = 'b')
plt.fill_between(t, q_l_x, q_u_x, color = 'b', alpha =0.2)
plt.plot(np.linspace(0,1, len(data_test)), color = 'r')
plt.ylabel('x')
plt.savefig(os.path.join(root_folder, 'state_estim.png'))
plt.show(block=False)

plt.figure()
plt.plot(t, mean_p, color = 'b')
plt.fill_between(t, q_l_p, q_u_p, color = 'b', alpha =0.2)
plt.axhline(p_gt, color = 'r')
plt.ylabel('p')
plt.ylim(0, max(damage_rate)*2)
plt.savefig(os.path.join(root_folder, 'param_estim.png'))
plt.show()
