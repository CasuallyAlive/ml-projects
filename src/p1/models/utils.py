import os, glob, re
import numpy as np
import pandas as pd
from typing import Union
from word2number.w2n import word_to_num

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

def get_data_col(data: pd.DataFrame, colname:str):
    x = data.loc[:, data.columns.isin([colname])]
    return x.to_numpy()
    
def get_common_label(y):
        p = len(np.where(y)[0])
        p_ = len(np.where(~y)[0])
        
        cl = p > p_
        return cl, p if cl else p_

def get_num(words):
    l=len(words)
    r = re.compile(r'teenth')
    for w in words:
        if(str.isdigit(w)):
            return int(w)
        elif(r.search(w)):
            return word_to_num(w[:-2])
        elif(str.isdigit(w.strip('a'))):
            return int(w.strip('a'))
        else:
            try:
                num=word_to_num(w)
            except:
                continue
            return num
    return False    

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

def get_attr_nums(data: pd.DataFrame, attr_id: str, uk:str):
    x = get_data_col(data, attr_id)

    dlen = len(x)

    nums = []

    non_ages_set = set(); non_ages_set.add(uk)
    for i in range(dlen):
        a = x[i][0].strip(' ')
        # print(a)
        if a == uk: nums.append(np.nan)
        elif str.isdigit(a): 
            num = int(a)        
            nums.append(num)
        elif get_num(re.sub("[^\w]", " ", a).split()):
            num = get_num(re.sub("[^\w]", " ", a).split())        
            nums.append(num)
        else: 
            nums.append(np.nan)
            non_ages_set.add(a)

    nums = np.array(nums)
    return nums


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

def get_project_data(opts=[True, True, True, True], data_dir="../project_data/data"):
    '''
    Order: test, eval, train
    Data Tuple: bow, glove, tfidf, misc
    '''
    data_dir_dict = dict()
    for folder in os.listdir(data_dir):
        # print(folder)
        data_dir_dict[folder] = list()
        for file_name in glob.glob(f"{data_dir}/{folder}/*.csv"):
            # prprint(bag_of_words_k_folds)int(file_name)
            data_dir_dict[folder].append(os.path.abspath("./") + "/" + file_name)
            # print(os.path.abspath("./") + "/" + file_name)

    # train
    # Load data and labels
    Y = []
    for i in range(3):
        t = pd.read_csv(data_dir_dict["glove"][i])
        Y.append((t.loc[:, t.columns.isin(['label'])].to_numpy() == 1).flatten())
        
    bow, glove, tfidf, misc = [], [], [], []
    for i in range(3):
        if(opts[0]):
            bow.append(pd.read_csv(data_dir_dict["bag-of-words"][i]))
        if(opts[1]):
            glove.append(pd.read_csv(data_dir_dict["glove"][i]))
        if(opts[2]):
            tfidf.append(pd.read_csv(data_dir_dict["tfidf"][i]))
        if(opts[3]):
            misc.append(pd.read_csv(data_dir_dict["misc"][-(i+1)]))
    return (bow, glove, tfidf, misc), Y
    
# Runs five fold CV on model.
# def five_fold_CV(k_datasets, labels, hyperparams, e, update_fnc, lr_fnc):
#     K=5
#     cv_acc = {tuple(hp) : [] for hp in hyperparams}
#     for r, mu in hyperparams:
#         for k in range(K):
#             k_ds = list(k_datasets)

#             # Derive validation set
#             data_val = k_ds[k]
#             x_val, y_val = get_data_comp(data_val, labels=list(labels.values()))
            
#             k_ds.pop(k)

#             # Derive training set
#             data_train = pd.concat(k_ds)
#             x_train, y_train = get_data_comp(data_train, labels=list(labels.values()))

#             pn = Perceptron(labels, update_fnc, lr_fnc)
#             pn.train(data_train, epochs=e, r=r, mu=mu)

#             cv_acc[(r, mu)].append(pn.calc_acc(x_val, y_val))
    
#     # average across all folds
#     cv_acc_stats = {tuple(hp) : (np.mean(trials), np.std(trials)) for hp, trials in cv_acc.items()}

#     return cv_acc_stats, cv_acc