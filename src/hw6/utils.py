import os, glob, re
import numpy as np
import pandas as pd
from typing import Union
from svm import SVM

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1(p, r):
    return 2*((p*r) / (p+r))

def get_prediction_errors(y, y_pred):
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    
    return tp, fp, fn, tn

def load_data(labels, null='?', dir=r"data/"):
    data = pd.read_csv(dir)
    
    x = fill_null(data.loc[:, ~data.columns.isin(['label'])], null)
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x.to_numpy(), y.flatten(), pd.concat([data.loc[:, data.columns.isin(['label'])], x], axis=1)

def calc_acc(y_h, y):
    matches = np.sum(y_h == y)
    return matches/len(y_h)

def get_data_comp(data, labels, null='?'):
    data = data.copy()
    x = fill_null(data.loc[:, ~data.columns.isin(['label'])], null)
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x, y.flatten()

def get_data_col(data: pd.DataFrame, colname:str):
    x = data.loc[:, data.columns.isin([colname])]
    return x.to_numpy()
    
def get_common_label(y):
        p = len(np.where(y)[0])
        p_ = len(np.where(~y)[0])
        
        cl = p > p_
        return cl, p if cl else p_   

def normalize(x, axis=1):
    return x/np.max(x, axis=axis)

def get_attribute_dict(features: pd.DataFrame, missing_names):
    A = list(features.columns)
    a_dict = {a : np.unique(features.loc[:, features.columns.isin([a])].to_numpy()).tolist() for a in A}
    for a_name, val in missing_names:
        a_dict[a_name].append(val)
        
    return a_dict

def get_null_columns(x: pd.DataFrame, null='?'):
    temp = (x == null).idxmax(axis=0)
    return list(temp.index[temp > 0])


# Get all hyperparameter combinations
def get_hp_combs(lrs, mus):
    hyperparameters = np.array(np.meshgrid(lrs, mus, indexing='ij')).T
    m, n, _ = hyperparameters.shape
    return hyperparameters.reshape((m*n, 2))

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

def get_divided_data(data: pd.DataFrame, k = 5):
    n = int(len(data) / k)
    list_df = [data[i:i+n] for i in range(0,len(data), n)]
    return list_df

def creat_pred_file(pred: np.ndarray, filename):
    pred_dict = {"example_id" : range(len(pred)), "label" : pred}
    pred_df = pd.DataFrame(pred_dict)
    
    pred_df.to_csv(filename, index=False)
    
# Runs five fold CV on model.
def five_fold_CV(k_datasets, labels, hyperparams, epochs):
    K=5
    cv_acc = {tuple(hp) : [] for hp in hyperparams}
    for r, C in hyperparams:
        for k in range(K):
            k_ds = list(k_datasets)

            # Derive validation set
            x_val, y_val = k_ds[k]            
            k_ds.pop(k)

            # Derive training set
            X = [k_ds[i][0] for i in range(len(k_ds))]
            Y = [k_ds[i][1] for i in range(len(k_ds))]
            
            x_train = np.concatenate(X)
            y_train = np.concatenate(Y)

            model = SVM(labels, r=r, C=C, epochs=epochs, suppress=True)
            model.train(x_train, y_train)

            cv_acc[(r, C)].append(calc_acc(model.predict(x_val), y_val))
    
    # average across all folds
    cv_acc_stats = {tuple(hp) : (np.mean(trials), np.std(trials)) for hp, trials in cv_acc.items()}

    return cv_acc_stats, cv_acc