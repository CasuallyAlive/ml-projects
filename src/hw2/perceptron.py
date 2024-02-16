import numpy as np
import pandas as pd
from typing import Union

def get_data_comp(data, labels):
    data = data.copy()
    x = data.loc[:, ~data.columns.isin(['label'])]
    y = data.loc[:, data.columns.isin(['label'])].to_numpy() == labels[0]
    
    return x, y.flatten()

# Bias term x_o in first column of feature matrix
def insert_bias_term(x: Union[pd.DataFrame, np.ndarray], m):
    
    t = x.copy().to_numpy() if type(x) is pd.DataFrame else x.copy()
    t = np.insert(t, 0, (np.zeros(shape=(m,))+1), axis=1)
    return t

def normalize(x, axis=0):
    return x/np.max(x, axis=axis)

class Perceptron:    
    
    # Baseline Perceptron update function
    def __baseline_update__(self, x_i, y_i, p, r):
        if p <= 0:
            self._w += r*x_i*self.labels[y_i]
            self.update_count += 1
    
    # Margin Perceptron
    def __margin_update__(self, x_i, y_i, p, r):
        if p < self._mu:
            self._w += r*x_i*self.labels[y_i]
            self.update_count += 1
    
    # Averaged Perceptron update function
    def __averaged_update__(self, x_i, y_i, p, r):
        if p <= 0:
            self._w += r*x_i*self.labels[y_i]
            self.update_count += 1
        self._a += self._w
        
    # Aggressive Perceptron w/ Margin update function
    def __aggr_margin_update__(self, x_i, y_i, p, r):
        if p <= self._mu:
            self._w += r*x_i*self.labels[y_i]
            self.update_count += 1
        
    # learning rate functions    
    lr_base_fnc = lambda r, mu, t, p, x : r
    lr_decay_fnc = lambda r, mu, t, p, x : r/(t+1.0)
    lr_opt_fnc = lambda r, mu, t, p, x : ((mu - p) / (np.dot(x.T, x)+1)).flatten()[0]
    
    # weight update functions
    update_base_fnc = __baseline_update__
    update_margin_fnc = __margin_update__
    update_averaged_fnc = __averaged_update__
    update_aggr_margin_fnc = __aggr_margin_update__
    
    def __init__(self, labels: dict, update_fnc=update_base_fnc, lr_fnc = lr_base_fnc, model=(None, None)):
        self._w = model[0]
        self._a = model[1]
        
        self.update_count = 0
        
        self._rand =  np.random.default_rng()
        
        self.labels = labels
        self._labels = {v: k for k, v in labels.items()}
        
        # Model functions
        self.update_fnc = update_fnc
        self.lr_fnc = lr_fnc
        
    # Iteratively train the model on examples and output pairs.
    def __train__(self, x:pd.DataFrame, y:pd.DataFrame, t=0):
        m, n = x.shape
        
        if self._w is None:
            self._w = self._rand.uniform(low=-0.01, high=0.01, size=(n+1,1))
            self._a = self._w.copy()  
                  
        examples = insert_bias_term(x, m)
        for i, example in enumerate(examples):
            e = example.reshape(self._w.shape)
            t+=1
            
            pred = self.labels[y[i]]*np.dot(self._w.T, e).flatten()[0]
            r = self.lr_fnc(self._r, self._mu, t, pred, e)
            
            self.update_fnc(self, e, y[i], pred, r)
            
        return t
    
    # Calculate classifier accuracy.
    def calc_acc(self, x, y, norm=False):
        y_h = self.predict(x, norm=norm)
        matches = np.sum(y_h == y)
        return matches/len(y_h)
    
    # Standard Perceptron training algorithm w/ epochs
    def train(self, dataset:pd.DataFrame, epochs=1, r=1.0, mu=1.0, dev_data = None, norm=False):
        m, n = dataset.shape
        t=0
        
        if self._w is None:
            self._w = self._rand.uniform(low=-0.01, high=0.01, size=(n,1))
            self._a = self._w.copy()
        
        self._r = r; self._mu=mu
        
        self.update_count = 0
        
        train_acc_ls = {}
        x_dev, y_dev = None, None
        if dev_data is not None:
            x_dev, y_dev = get_data_comp(dev_data.copy(), labels=list(self.labels.values()))
            x_dev = normalize(x_dev) if norm else x_dev
        
        for e in range(epochs):
            # Shuffle dataset
            data = dataset.copy().sample(frac=1)
            x_e, y_e = get_data_comp(data.copy(), labels=list(self.labels.values()))
            x_e = normalize(x_e) if norm else x_e
            
            # Train model
            t=self.__train__(x_e, y_e, t=t)
            
            if dev_data is not None:
                acc = self.calc_acc(x_dev, y_dev)
                classifier = self._w.copy() if self.update_fnc is not Perceptron.update_averaged_fnc else self._a.copy()
                train_acc_ls[e] = (acc, classifier)
        
        return self.update_count, train_acc_ls
        
    # Predict ezample(s). Inserts bias term if necessary
    def predict(self, x: Union[pd.DataFrame, np.ndarray], norm=False):
        if(self._w is None):
            return
        if(type(x) is pd.DataFrame):
            x = x.copy().to_numpy()
        if(len(x.shape) == 1):    
            x = x.reshape((x.shape[0], 1))
        m, n = x.shape
        
        x = normalize(x) if norm else x
        if(m != n and n == self._w.shape[0] - 1):
            x = insert_bias_term(x, m).T
        
        out = np.dot(self._w.T, x).flatten() if self.update_fnc is not Perceptron.update_averaged_fnc \
                    else np.dot(self._a.T, x).flatten()
        
        return out >= 0
    