import numpy as np
import pandas as pd

def load_data(labels, null='?', dir=r"data/"):
    data = pd.read_csv(dir)
    
    x = fill_null(data.loc[:, ~data.columns.isin(['label'])], null)
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x, y.flatten(), pd.concat([data.loc[:, data.columns.isin(['label'])], x], axis=1)

def calc_acc(y_h, y):
    matches = np.sum(y_h == y)
    return matches/len(y_h)

def get_attribute_dict(features: pd.DataFrame, missing_names):
    A = list(features.columns)
    a_dict = {a : np.unique(features.loc[:, features.columns.isin([a])].to_numpy()).tolist() for a in A}
    for a_name, val in missing_names:
        a_dict[a_name].append(val)
        
    return a_dict

def get_null_columns(x: pd.DataFrame, null='?'):
    temp = (x == null).idxmax(axis=0)
    return list(temp.index[temp > 0])

def fill_null(x:pd.DataFrame, null='?'):
    col_names = get_null_columns(x, null)

    for col_name in col_names:
        col_missing = x.loc[:, x.columns.isin([col_name])]
        col_mode = col_missing.mode()[col_name][0]
        
        x.loc[:, x.columns == col_name] = col_missing.mask(col_missing == null, other=col_mode)  
    return x