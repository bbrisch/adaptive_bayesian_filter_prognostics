#%% TODO: convert to ipynb
%load_ext autoreload
%autoreload 2
%matplotlib inline



import yaml

import os
import pickle as pkl

import src
from src.plotting_fns import get_bounds
from src.bayesian_opt import compute_loss
import matplotlib.pyplot as plt


from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances, plot_slice




#%%

path = os.path.join('results', 'test_crps_long2.pkl')

with open(path, 'rb') as f:
    result = pkl.load(f )
config, study = result
# plot_optimization_history(study)
# plt.show()

# plot_parallel_coordinate(study)
# plt.show()
plot_param_importances(study)
plot_optimization_history(study)
plot_slice(study)
# plt.show()



#%%

best_params = study.best_params
print(config["dataloader_config"])
validation = src.ValidationObj(config,best_params,'train')
gt_x, gt_param,rul_gt,results = validation()
# print(compute_loss(gt_x, gt_param,rul_gt,results ))

src.plot_hi(gt_x,results)

src.plot_param(gt_param, results)
src.plot_rul(rul_gt, results)
plt.show()

print(best_params)

# %% Compare both studies

#1. open both studies

# %%
