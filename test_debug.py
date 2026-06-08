#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline


import os
import yaml
import matplotlib.pyplot as plt

from src.utils.dataloader import train_test_split
from src.bayesian_opt import initialize_data,initialize_filter, sample_parameters, compute_predictions, compute_loss
from src.plotting_fns import plot_rul, plot_hi, plot_param, get_bounds

#%%
with open(os.path.join('src','sweep_config.yml'), 'rb') as f:
    config = yaml.safe_load(f)

# train_test_split(config['dataloader_config']['path_data'])
fold = 'special'
config['dataloader_config']['path_data'] = os.path.join(config['dataloader_config']['path_data'], fold)

#%%
#  get the hyper parameters from the trial object
param_dict = sample_parameters(config,None)

# Initialization
data_init = initialize_data(config["dataloader_config"])
particle_filter, dataloader, initial_state = initialize_filter(config, param_dict, data_init)

#%%
# Compute predictions
gt_x,gt_param,rul_gt,results = compute_predictions( particle_filter, dataloader, initial_state)

# %%

plot_rul(rul_gt,results)
plot_hi(gt_x, results)
plot_param(gt_param, results)
plt.show()
print(compute_loss(gt_x,gt_param,rul_gt,results, 'crps'))
print(compute_loss(gt_x,gt_param,rul_gt,results, 'mse'))


# %%
get_bounds(results[0].get_x(), results[0].get_w(), 0.75)
# %%
