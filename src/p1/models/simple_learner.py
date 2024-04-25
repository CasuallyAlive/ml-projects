import pandas as pd 
import numpy as np
from .utils import get_data_col, get_num

class SL:
    
    def train(self, x: pd.DataFrame, y:np.ndarray, labels: dict):
        self.labels = labels
        pass
    
    def predict(self, x: np.ndarray):
        predictions = np.zeros(shape=(x.shape[0],))
        for i, xi in enumerate(x):
            if np.isnan(xi):
                predictions[i] = 1
            else:
                predictions[i] = 0
                
        return predictions
        