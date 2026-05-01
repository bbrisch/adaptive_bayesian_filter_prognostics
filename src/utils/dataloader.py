import os

import pandas as pd
import numpy as np
import pickle as pkl



class Dataloader():
    def __init__(self, path_data, hi = 'percentual', thresh = np.inf, exclude_ids = [3,4]):
        self.path_data = path_data
        self.hi = hi
        self.thresh  =thresh
        self.exclude_ids =exclude_ids
        
        # get list of folders
        folders = os.listdir(path_data)
        self.data_folder = dict()
        for f in folders:
            if 'DIC' in f:
                
                id = int(f[f.find('.csv')-2:f.find('.csv')])
                if id in exclude_ids:
                    continue
                self.data_folder.update({id:f})
                
        self.ids = list(self.data_folder.keys())
        
    def _get_health_indicator(self, df):
        data =  df.strain.values
        if self.hi == 'none':
            return data[data <= self.thresh]
        
        elif self.hi == 'percentual':
            s_0 = data[0]
            data = (data-s_0)/s_0
            return data[data <= self.thresh]
        elif self.hi == 'stiffness_red':
            s_0 = data[0]
            data = 1-s_0/data
            return data[data <= self.thresh]
    
    def load_id(self, id):
        df = pd.read_csv(os.path.join(self.path_data, self.data_folder[id]), names=['cycle', 'strain'])
        hi = self._get_health_indicator(df)
        return hi
    
    def load_id_set(self, id_list):
        return [self.load_id(id) for id in id_list]
    
            

        
    
    
    