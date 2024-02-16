import numpy as np
import pandas as pd
from typing import Union
from utils import get_data_comp, insert_bias_term

class Perceptron:    
    
    # Baseline Perceptron update function
    def __baseline_update__(self, x_i, y_i, p, r):
        if p <= 0:
            self._w += r*x_i*self.labels[y_i]
    
    # Margin Perceptron
    def __margin_update__(self, x_i, y_i, p, r):
        if p < self._mu:
            self._w += r*x_i*self.labels[y_i]
    
    # Averaged Perceptron update function
    def __averaged_update__(self, x_i, y_i, p, r):
        if p <= 0:
            self._w += r*x_i*self.labels[y_i]
        self._a += self._w
        
    # Aggressive Perceptron w/ Margin update function
    def __aggr_margin_update__(self, x_i, y_i, p, r):
        if p <= self._mu:
            self._w += r*x_i*self.labels[y_i]
        
    # learning rate functions    
    lr_base_fnc = lambda r, mu, t, p, x : r
    lr_decay_fnc = lambda r, mu, t, p, x : r/(t+1.0)
    lr_opt_fnc = lambda r, mu, t, p, x : ((mu - p) / (np.dot(x.T, x)+1))
    
    # weight update functions
    update_base_fnc = __baseline_update__
    update_margin_fnc = __margin_update__
    update_averaged_fnc = __averaged_update__
    update_aggr_averaged_fnc = __aggr_margin_update__
    
    def __init__(self, labels: dict, update_fnc=update_base_fnc, lr_fnc = lr_base_fnc):
        self._w = None
        self._a = None
        
        self._rand =  np.random.default_rng()
        
        self.labels = labels
        self._labels = {v: k for k, v in labels.items()}
        
        # Model functions
        self.update_fnc = update_fnc
        self.lr_fnc = lr_fnc
        
        # Selected functions for use during training
        self.update_lr_func = self.__train__
        self.error_func = self.__train__
        self.margin = self.__train_margin__
        self.averaged = self.__train_average__
    
    # Iteratively train the model on examples and output pairs.
    def __train__(self, x:pd.DataFrame, y:pd.DataFrame, t=0):
        m, n = x.shape
        
        if self._w is None:
            self._w = self._rand.uniform(low=-0.01, high=0.01, size=(n+1,1))   
                  
        examples = insert_bias_term(x, m)
        for i, example in enumerate(examples):
            
            r = self.lr_fnc(self._r, self._mu, t)
            
            self.update_fnc(example.reshape(self._w.shape), y[i], r)
            # if y[i] != h:
            #     self.__baseline_update__(example.reshape(self._w.shape), y[i], r=r_d)
            
        return t
        
    # Standard Perceptron training algorithm w/ epochs
    def train(self, dataset:pd.DataFrame, epochs=1, r=1.0, mu=1.0):
        m, n = dataset.shape
        t=0
        self._w = self._rand.uniform(low=-0.01, high=0.01, size=(n,1))
        self._r = 1.0; self._mu=1.0
        
        for e in range(epochs):
            # Shuffle dataset
            data = dataset.copy().sample(frac=1)
            x_e, y_e = get_data_comp(data.copy(), labels=list(self.labels.values()))
            
            # Train model
            t=self.__train__(x_e, y_e, t=t)
        
        return
        
    # Predict ezample(s). Inserts bias term if necessary
    def predict(self, x: Union[pd.DataFrame, np.ndarray]):
        if(self._w is None):
            return
        if(type(x) is pd.DataFrame):
            x = x.copy().to_numpy()
        if(len(x.shape) == 1):    
            x = x.reshape((x.shape[0], 1))
        m, n = x.shape
        if(m != n and n == self._w.shape[0] - 1):
            x = insert_bias_term(x, m).T

        out = np.dot(self._w.T, x).flatten()
        return out >= 0
    
