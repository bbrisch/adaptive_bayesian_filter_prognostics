import os
import shutil
import optuna
import numpy as np 
import pickle as pkl
from  scipy.optimize import curve_fit

from .utils import Dataloader

def train_test_split(data_path,ratio = 0.8, seed = 42):
    
    # If the training already exists, dont do anything
    if os.path.exists(os.path.join(data_path, 'test')):
        return None
    
    # Other case:
    files_standard = []
    files_special = []
    for f in os.listdir(data_path):
        
        if not ('.csv' in f and 'DIC' in f):
            continue
        end_pos = f.find('.csv')
        file_id = int(f[end_pos-2:end_pos])
        print(f, file_id)
        
        
        # Separate the special cases
        if file_id in [9,10,11,12]:
            files_special.append(f)
        
        else: 
            files_standard.append(f)
    
    # Choose the training subset
    np.random.seed(seed)
    files_train = np.random.choice(files_standard, int(len(files_standard)*ratio))
    
    os.makedirs(os.path.join(data_path, 'train'))
    os.makedirs(os.path.join(data_path, 'test'))
    os.makedirs(os.path.join(data_path, 'special'))
    
    for f in os.listdir(data_path):
        if not ('.csv' in f and 'DIC' in f):
            continue
        if f in files_special:
            shutil.copy(os.path.join(data_path,f), os.path.join(data_path,'special',f))
        elif f in files_train:
            shutil.copy(os.path.join(data_path,f), os.path.join(data_path,'train',f))
        elif not f in files_train:
            shutil.copy(os.path.join(data_path,f), os.path.join(data_path,'test',f))
            
              
        
    
    


# Obs eq.
def poli_fn(hi, a,b,c):
        return a*hi**3+b*hi**2+c*hi
    

def initialize_objects(dataloader_conf):
    
    dataloader_conf['path_data'] = os.path.join(dataloader_conf['path_data'], 'train')
    
    # Instance the dataloader
    dataloader = Dataloader(**dataloader_conf)
    
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
        errors.append(y-y_)
    obs_std = np.std(errors)
    
    return dataloader, obs_std, popt



def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return (x - 2) ** 2


def optimize_params():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    found_x = best_params["x"]
    found_y = best_params["y"]
    
    print(f"Found x: {found_x}, Found y: {found_y}, (x - 2)^2: {(found_x - 2) ** 2}")
    return study
    



    
    
        
    