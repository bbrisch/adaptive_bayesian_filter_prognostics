import os
import shutil
import pandas as pd
import numpy as np
import pickle as pkl


class Dataloader:
    def __init__(self, path_data, hi="percentual", thresh=np.inf, exclude_ids=[2,3,4]):
        self.path_data = path_data
        self.hi = hi
        self.thresh = thresh
        self.exclude_ids = exclude_ids

        # get list of folders
        folders = os.listdir(path_data)
        self.data_folder = dict()
        for f in folders:
            if "DIC" in f:

                id = int(f[f.find(".csv") - 2 : f.find(".csv")])

                if id in exclude_ids:
                    continue
                self.data_folder.update({id: f})

        self.ids = list(self.data_folder.keys())

    def _get_health_indicator(self, df):
        data = df.strain.values
        if self.hi == "none":
            return data[data <= self.thresh]

        elif self.hi == "percentual":
            s_0 = data[0]
            data = (data - s_0) / s_0
            return data[data <= self.thresh]
        elif self.hi == "stiffness_red":
            s_0 = data[0]
            data = 1 - s_0 / data
            return data[data <= self.thresh]

    def load_id(self, id):
        df = pd.read_csv(
            os.path.join(self.path_data, self.data_folder[id]),
            names=["cycle", "strain"],
        )
        hi = self._get_health_indicator(df)
        return hi,1/len(hi)

    def load_id_set(self, id_list):
        return [self.load_id(id)[0] for id in id_list], [self.load_id(id)[1] for id in id_list]
    
    def get_norm_const(self):
        x,p = self.load_id_set(self.ids)
        
        x = np.concatenate(x)
        p = np.array(p)
        
        return x.mean(), x.std(0),p.mean(),p.std(0)
        


def train_test_split(data_path, ratio=0.8, seed=42, ids_exlude = [2,3,4]):

    # If the training already exists, dont do anything
    if os.path.exists(os.path.join(data_path, "test")):
        return None

    # Other case:
    files_standard = []
    files_special = []
    for f in os.listdir(data_path):

        if not (".csv" in f and "DIC" in f):
            continue
        end_pos = f.find(".csv")
        file_id = int(f[end_pos - 2 : end_pos])
        print(f, file_id)

        # Separate the special cases
        if file_id in ids_exlude:
            continue
        elif file_id in [9, 10, 11, 12]:
            files_special.append(f)

        else:
            files_standard.append(f)

    # Choose the training subset
    np.random.seed(seed)
    files_train = np.random.choice(files_standard, int(len(files_standard) * ratio))

    os.makedirs(os.path.join(data_path, "train"))
    os.makedirs(os.path.join(data_path, "test"))
    os.makedirs(os.path.join(data_path, "special"))

    for f in os.listdir(data_path):
        
        
        if not (".csv" in f and "DIC" in f):
            continue
        
        end_pos = f.find(".csv")
        file_id = int(f[end_pos - 2 : end_pos])
        if file_id in ids_exlude:
            continue
        elif f in files_special:
            shutil.copy(
                os.path.join(data_path, f), os.path.join(data_path, "special", f)
            )
        elif f in files_train:
            shutil.copy(os.path.join(data_path, f), os.path.join(data_path, "train", f))
        elif not f in files_train:
            shutil.copy(os.path.join(data_path, f), os.path.join(data_path, "test", f))

def train_special_split(data_path, ids_exlude = [2,3,4]):
    try:
        os.makedirs(os.path.join(data_path, "train"))
        os.makedirs(os.path.join(data_path, "test"))
        os.makedirs(os.path.join(data_path, "special"))
    except:
        return 
        
    for f in os.listdir(data_path):
         
        if not (".csv" in f and "DIC" in f):
            continue
        end_pos = f.find(".csv")
        file_id = int(f[end_pos - 2 : end_pos])
        print(f, file_id)
    
        if file_id in ids_exlude:
            continue
        elif file_id in [9, 10, 11, 12]:
            shutil.copy(
                os.path.join(data_path, f), os.path.join(data_path, "special", f)
            )
        else:
            shutil.copy(os.path.join(data_path, f), os.path.join(data_path, "train", f))
    