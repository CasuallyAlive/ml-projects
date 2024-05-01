import numpy as np

class SVM:
    def __init__(self, labels, r=0.01, C=1.0, tol=1e-4, epochs=100, suppress=False):
        self.r = r
        self.C = C  # regularization parameter
        self.epochs = epochs  # number of epochs
        self.labels = labels
        self.tol = tol
        self.suppress = suppress
        
        self._w = None
        self._rand =  np.random.default_rng()
    
    def _svg(self, X:np.ndarray, y:np.ndarray, m, r):        
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        # Update parameters using stochastic sub-gradient descent
        for i, xi in enumerate(X):
            xi = xi.reshape(self._w.shape)
            
            pred = self._sgn(y[i]) * self._pred_ex(xi)
            
            # Compute the sub-gradient of the hinge loss
            self._w = (1 - r)*self._w
            if pred <= 1.0:
                self._w += r * self.C * self._sgn(y[i]) * xi
        
        return
    
    def _pred_ex(self, xi):
        return np.dot(self._w.T, xi).squeeze()
    
    def _sgn(self, y):
        if isinstance(y, np.ndarray) and len(y) > 1:
            return np.array([self._sgn(yi) for yi in y])
        return self.labels[int(y)]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        t=0
        
        self._w = self._rand.uniform(low=-1e-5, high=1e5, size=(n+1,1))

        dataset = np.column_stack((y, X))
        
        loss = np.zeros(self.epochs)
        loss = np.insert(loss, 0, float('inf'))
        
        for t in range(self.epochs):
            # Shuffle dataset
            data = dataset.copy()
            self._rand.shuffle(x=data, axis=0)
            
            _y, _X = data[:, 0], data[:, 1:]
            
            r = self.r / (1 + t)
            self._svg(X=_X, y=_y, m=m, r=r)
        
            # Compute the total hinge loss for the epoch
            total_loss = self.calc_loss(_X, _y)

            if np.abs(total_loss - loss[t]) < self.tol:
                if(not self.suppress):
                    print(f"Converged at epoch T={t}")
                break
            
            loss[t+1] = total_loss
                    
        return loss[(loss < float('inf')) & (loss > 0.0)]

    def calc_loss(self, X, y):
        return np.sum(np.maximum(0, 1 - self._sgn(y) * self._sgn(self.predict(X))))
    
    def predict(self, X: np.ndarray):
        if(self._w is None):
            return
        
        m, n = X.shape
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        out = np.sign(np.dot(X, self._w)).flatten()
        
        return out >= 0