import numpy as np
import pandas as pd
from typing import Union
from perceptron import Perceptron

def load_data(labels, null='?', dir=r"data/"):
    data = pd.read_csv(dir)
    
    x = fill_null(data.loc[:, ~data.columns.isin(['label'])], null)
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x, y.flatten(), pd.concat([data.loc[:, data.columns.isin(['label'])], x], axis=1)

def calc_acc(y_h, y):
    matches = np.sum(y_h == y)
    return matches/len(y_h)

def get_data_comp(data, labels, null='?'):
    data = data.copy()
    x = fill_null(data.loc[:, ~data.columns.isin(['label'])], null)
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x, y.flatten()

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

 # Bias term x_o in first column of feature matrix
def insert_bias_term(x: Union[pd.DataFrame, np.ndarray], m):
    
    t = x.copy().to_numpy() if type(x) is pd.DataFrame else x.copy()
    t = np.insert(t, 0, (np.zeros(shape=(m,))+1), axis=1)
    return t

def five_fold_CV(k_datasets, labels, learning_rates, options):
    K=5
    cv_acc = {r : [] for r in learning_rates}
    for r in learning_rates:
        for k in range(K):
            k_ds = list(k_datasets)

            # Derive validation set
            data_val = k_ds[k]
            x_val, y_val = get_data_comp(data_val, labels=list(labels.values()))
            
            k_ds.pop(k)

            # Derive training set
            data_train = pd.concat(k_ds)
            x_train, y_train = get_data_comp(data_train, labels=list(labels.values()))

            pn = Perceptron(labels)
            pn.train(data_train, epochs=options[0], r=r, decay=options[2])

            cv_acc[r].append(calc_acc(pn.predict(x_val), y_val))
            
    return cv_acc