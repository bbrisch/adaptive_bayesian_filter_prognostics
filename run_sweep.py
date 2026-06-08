import os
import yaml
import pickle as pkl

import src
import matplotlib.pyplot as plt


from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances

with open(os.path.join("src", "sweep_config.yml"), "rb") as f:
    config = yaml.safe_load(f)

# src.train_test_split(config['dataloader_config']['path_data'], 1)
src.train_special_split(config['dataloader_config']['path_data'])
os.makedirs('results', exist_ok=True)



study = src.optimize_params(config)
best_params = study.best_params

with open(os.path.join('results', config['name']+'.pkl'), "wb") as f:
    pkl.dump((config,study), f)
