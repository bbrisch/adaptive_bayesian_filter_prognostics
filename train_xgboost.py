import os
import yaml
from src import xgboost


with open(os.path.join("src", "sweep_config.yml"), "rb") as f:
    config = yaml.safe_load(f)

obj = xgboost.XgboostModel(config)
obj.train()
obj.save('xgb')
# preds, y_series = obj.eval('train')

# import matplotlib.pyplot as plt

# for p,y in zip(preds, y_series):
#     plt.figure()
#     plt.plot(p)
#     plt.plot(y, color = 'r')
#     plt.show()
    
    
    
# preds, y_series = obj.eval('special')

# import matplotlib.pyplot as plt

# for p,y in zip(preds, y_series):
#     plt.figure()
#     plt.title('special')
#     plt.plot(p)
#     plt.plot(y, color = 'r')
# plt.show()