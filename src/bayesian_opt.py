import optuna
import numpy as np 
from  scipy.optimize import curve_fit

import utils


def initialize_objects(dataloader_conf):
    
    # Instance the dataloader
    dataloader = utils.Dataloader(dataloader_conf)
    
    # Fit obs_eq
    def poli_fn(hi, a,b,c):
        return a*hi**3+b*hi**2+c*hi
    
    # Adjust the values of the measurement eq.
    data_train = []
    hi_train = []
    lengths = []
    damage_rate = []
    for f in dataloader.ids:
        data = dataloader.load_id(f)
        
        lengths.append(len(data))
        damage_rate.append(1/len(data))
        
        data_train.extend(data)
        hi_train.extend(np.linspace(0,1, len(data)))
        
    msg= curve_fit(poli_fn, hi_train,data_train,p0=(1,1,1), full_output=False)
    popt = msg[0]
    
    # get noise level and damage accumulation rates    
    errors = []
    for y, h in zip(data_train, hi_train):
        y_ = poli_fn(h, *popt)
        errors.extend(y-y_)
    obs_std = np.std(errors)
    
    return dataloader, obs_std, popt
    
    
    
    
    
    
        
    