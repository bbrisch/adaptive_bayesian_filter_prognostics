#%%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


import os
import yaml
from src.bayesian_opt import initialize_filter, sample_parameters, compute_predictions, compute_loss
from src.plotting_fn import plot_rul, plot_hi, plot_param

#%%
with open(os.path.join('src','sweep_config.yml'), 'rb') as f:
    config = yaml.safe_load(f)


fold = 'train'
config['dataloader_config']['path_data'] = os.path.join(config['dataloader_config']['path_data'], fold)

#%%
#  get the hyper parameters from the trial object
param_dict = sample_parameters(config,None)

# Initialization
particle_filter, dataloader, initial_state = initialize_filter(config, param_dict)

#%%
# Compute predictions
gt_x,gt_param,rul_gt,results = compute_predictions( particle_filter, dataloader, initial_state)
print(compute_loss(gt_x,gt_param,rul_gt,results))
# %%

plot_rul(rul_gt,results)
plot_hi(gt_x, results)
plot_param(gt_param, results)


# %%
