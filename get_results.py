#%%
import yaml

import os
import pickle as pkl

import src
from src.bayesian_opt import compute_loss
import matplotlib.pyplot as plt


from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances, plot_slice




#%%

with open('study2.pkl', 'rb') as f:
    study = pkl.load(f )
    

#%%

# plot_optimization_history(study)
# plt.show()

# plot_parallel_coordinate(study)
# plt.show()
plot_param_importances(study)
plot_optimization_history(study)
plot_slice(study)
# plt.show()



#%%
with open(os.path.join('src','sweep_config.yml'), 'rb') as f:
    config = yaml.safe_load(f)

best_params = study.best_params

validation = src.ValidationObj(config,'test')
gt_x, gt_param,rul_gt,results = validation(best_params)
print(compute_loss(gt_x, gt_param,rul_gt,results ))

src.plot_hi(gt_x,results)

src.plot_param(gt_param, results)
src.plot_rul(rul_gt, results)
plt.show()

print(best_params)

# %%
