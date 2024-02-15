import numpy as np
import pandas as pd
from typing import Union

class Perceptron:
    
    def __init__(self, labels: dict):
        self._w = None
        self.labels = labels
        self._labels = {v: k for k, v in labels.items()}
    
    # Update weight vectors
    def __update__(self, x_i, y_i, r):
        self._w += r*x_i*self.labels[y_i]
    
    # Iteratively train the model
    def train(self, x:pd.DataFrame, y:pd.DataFrame, r=1.0):
        m, n = x.shape
        self._w = np.zeros(shape=(n+1,1))
        
        examples = x.to_numpy()
        examples = np.append(examples, (np.zeros(shape=(m, 1))+1), axis=1)
        
        for i, example in enumerate(examples):
            h = self.predict(x=example.reshape(self._w.shape))[0]
            if y[i] != h:
                self.__update__(example.reshape(self._w.shape), y[i], r)
        
        return
    
    # Predict ezample(s)
    def predict(self, x: Union[pd.DataFrame, np.ndarray]):
        
        if(self._w is None):
            return
        if(type(x) is pd.DataFrame):
            x = x.to_numpy()
        if(len(x.shape) == 1):    
            x = x.reshape((x.shape[0], 1))
        m, n = x.shape
        if(m != n and n == self._w.shape[0] - 1):
            x = np.append(x, (np.zeros(shape=(m, 1))+1), axis=1).T

        out = np.dot(self._w.T, x).flatten()
        return out >= 0
    
    