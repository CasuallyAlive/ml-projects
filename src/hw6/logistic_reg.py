import numpy as np

class LogisticRegression:
    
    def __init__(self, labels, r=0.01, sigma=1, tol=1e-4, epochs=100, suppress=False, exit_on_converge=True):
        self.r = r
        self.sigma = sigma if sigma != 0 else 1e-16
        self.epochs = epochs
        self.labels = labels
        self.suppress = suppress
        self.exit_on_converge = exit_on_converge
        self.tol=tol
        
        self._w = None
        self._rand = np.random.default_rng()
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _pred_ex(self, xi):
        return np.dot(xi.flatten(), self._w.flatten()).squeeze()
    
    def _sgd(self, X:np.ndarray, y:np.ndarray, m, r):        
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        # Update params using SGD
        for i, xi in enumerate(X):
            xi = xi.reshape(self._w.shape)
            yi = y[i]
            
            linear_out = self._pred_ex(xi)
            pred = self._sigmoid(linear_out)
            
            # Compute Gradient
            dw = xi * (pred - yi) + (1 / (self.sigma)**2) * self._w
            
            dw[0] = pred - yi
            
            # Update weights
            self._w -= r * dw
    
    def _sgn(self, y):
        if isinstance(y, np.ndarray) and len(y) > 1:
            return np.array([self._sgn(yi) for yi in y])
        return self.labels[int(y)]
    
    def train(self, X:np.ndarray, y: np.ndarray):
        m, n = X.shape
        t = 0
        
        self._w = self._rand.uniform(low=-1e-8, high=1e-8, size=(n+1,1))
        
        dataset = np.column_stack((y, X))
        
        loss = np.zeros(self.epochs)
        loss = np.insert(loss, 0, float('inf'))
        
        for t in range(self.epochs):
             # Shuffle dataset
            data = dataset.copy()
            self._rand.shuffle(x=data, axis=0)
            
            _y, _X = data[:, 0], data[:, 1:]
            
            r = self.r / (1 + t)
            self._sgd(X=_X, y=_y, m=m, r=r)
            
            # Compute the total loss for the epoch
            total_loss = self.calc_loss(_X, _y)
            
            if self.exit_on_converge and np.abs(total_loss - loss[t]) < self.tol:
                if(not self.suppress):
                    print(f"Converged at epoch T={t+1}")
                break
            
            loss[t+1] = total_loss
            
        return loss[1:]
    
    def calc_loss(self, X, y):
        
        m, _ = X.shape
        y_pred = self._predict(X)
        
        cross_entropy_loss = -(1 / m) * np.sum((y * np.log(y_pred + 1e-16) + (1 - y) * np.log(1 - y_pred + 1e-16)))
        reg_loss = (1 / (2 * self.sigma**2)) * np.sum(self._w**2)
        return cross_entropy_loss + reg_loss
    
    def _predict(self, X):
        m, n = X.shape
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        z = np.dot(X, self._w)
        return self._sigmoid(z).flatten()
    
    def predict(self, X: np.ndarray):
        if(self._w is None):
            return
        
        out = self._predict(X)
        return out >= 0.5