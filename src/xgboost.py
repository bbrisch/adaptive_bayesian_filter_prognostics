import xgboost
import copy
import os 
import numpy as np
import pickle as pkl
import optuna

from .utils import Dataloader
import matplotlib.pyplot as plt


def initialize_data(config):

    # Instance the dataloader
    dataloader = Dataloader(**config['dataloader_config'])

    # Get widows of a certain size
    input_series = []
    rul_series = []

    for f in dataloader.ids:
        data, rate = dataloader.load_id(f)
        rul = len(data)-np.arange(len(data))
        
        input_series.append(data)
        rul_series.append(rul)




    return input_series, rul_series

def sample_parameters(config, trial: optuna.trial.Trial):
    param_dict = dict()
    if config["xgboost"]["sample"]:
        for name, values in config["xgboost"]["search_space"].items():
            if values[-1] == "float":
                p = trial.suggest_float(name, values[0], values[1], log=values[2])
                param_dict.update({name: p})

            elif values[-1] == "int":
                p = trial.suggest_int(name, values[0], values[1], log=values[2])
                param_dict.update({name: p})

            else:
                assert (
                    False
                ), f"Param type not recognized\nname: {name}\nparams: {values}"
    else:
        for name, values in config["xgboost"]["manual_params"].items():
            param_dict.update({name: values})

    return param_dict


def initialize_model(config, param_dict, data):
    
    # Generate windows
    
    X = []
    y = []
    
    input_series, rul_series = data

    for x, rul in zip(input_series, rul_series):
        
        for i in range(param_dict['w_len'],len(x)):

            X.append(x[i-param_dict['w_len']:i])
            y.append(rul[i])
    
    X = np.array(X)
    y = np.array(y)
    
    xgb = xgboost.XGBRegressor(**param_dict)
    
    return X, y, xgb
    
def train_model(xgb:xgboost.XGBRegressor, X,y):
    xgb.fit(X,y)
    prd = xgb.predict(X)
    mse = (abs(prd-y)**2).mean()
    mae = abs(prd-y).mean()
    return xgb, mse, mae
    
    

def initialize_data_val(param_dict, data):
    
    # Generate windows
    x_series = []
    y_series = []
    
    input_series, rul_series = data

    for x, rul in zip(input_series, rul_series):
        
        X = []
        y = []
        for i in range(param_dict['w_len'],len(x)):

            X.append(x[i-param_dict['w_len']:i])
            y.append(rul[i])
    
        X = np.array(X)
        y = np.array(y)

        x_series.append(X)
        y_series.append(y)

    
    return x_series, y_series





class XgboostModel:
    def __init__(self, config):
        self.config_special = copy.deepcopy(config)
        self.config_special["dataloader_config"]["path_data"] = os.path.join(
            config["dataloader_config"]["path_data"], "special"
        )
        
        self.config = copy.deepcopy(config)
        self.config["dataloader_config"]["path_data"] = os.path.join(
            config["dataloader_config"]["path_data"], "train"
        )
        self.data = initialize_data(self.config)
        self.data_secial = initialize_data(self.config_special)

    
    def train(self):

        #  get the hyper parameters from the trial object
        param_dict = sample_parameters(self.config, None)

        # Initialization
        X,y, xgb = initialize_model(self.config, param_dict, self.data)
        xgb, mse, mae = train_model(xgb,X,y)
        print(mae)
        # return xgb, mse
        self.xgb = xgb
        
    def save(self, name):
        with open(os.path.join('results', name+'.pkl'), 'wb') as f:
            pkl.dump(self.xgb, f)
    
    def load(self, name):
        with open(os.path.join('results', name+'.pkl'), 'rb') as f:
            self.xgb = pkl.load(f)
            
    def eval(self, fold):
        if fold == 'train':
            data = self.data
        elif fold == 'special':
            data = self.data_secial
        else:
            assert False
        param_dict = sample_parameters(self.config, None)
        x_series, y_series  = initialize_data_val(param_dict, data)
        preds = []
        for X in x_series:
            preds.append(self.xgb.predict(X))
        return preds, y_series
        