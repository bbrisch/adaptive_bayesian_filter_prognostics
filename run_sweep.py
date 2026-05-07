import os
import yaml
import pickle as pkl

import src
import matplotlib.pyplot as plt


from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances


with open(os.path.join('src','sweep_config.yml'), 'rb') as f:
    config = yaml.safe_load(f)

src.train_test_split(config['dataloader_config']['path_data'])


dataloader, obs_std, popt  = src.initialize_objects(config['dataloader_config'])

study = src.optimize_params()

# with open('study.pkl', 'wb') as f:
#     pkl.dump(study,f )


plot_optimization_history(study)
plt.show()

plot_parallel_coordinate(study)
plt.show()
plot_param_importances(study)
plt.show()