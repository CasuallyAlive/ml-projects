import numpy as np

class SVM:
    def __init__(self, labels, r=0.01, C=1.0, tol=1e-4, epochs=100, suppress=False, exit_on_converge=True):
        self.r = r
        self.C = C  # regularization parameter
        self.epochs = epochs  # number of epochs
        self.labels = labels
        self.tol = tol
        self.suppress = suppress
        self.exit_on_converge = exit_on_converge
        
        self._w = None
        self._rand =  np.random.default_rng()
    
    def _sgd(self, X:np.ndarray, y:np.ndarray, m, r):        
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        # Update parameters using stochastic sub-gradient descent
        for i, xi in enumerate(X):
            xi = xi.reshape(self._w.shape)
            yi = y[i]
            
            pred = yi * self._pred_ex(xi)
            
            # Compute the sub-gradient of the hinge loss
            
            self._w = (1 - r) * self._w + r * self.C * yi * xi if pred <= 1 else \
                        (1 - r) * self._w
            
        return
    
    def _pred_ex(self, xi):
        return np.dot(xi.flatten(), self._w.flatten()).squeeze()
    
    def _sgn(self, y):
        if isinstance(y, np.ndarray) and len(y) > 1:
            return np.array([self._sgn(yi) for yi in y])
        return self.labels[int(y)]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        t=0
        
        self._w = self._rand.uniform(low=-1e-8, high=1e-8, size=(n+1,1))

        dataset = np.column_stack((y, X))
        
        loss = np.zeros(self.epochs)
        loss = np.insert(loss, 0, float('inf'))
        
        for t in range(self.epochs):
            # Shuffle dataset
            data = dataset.copy()
            self._rand.shuffle(x=data, axis=0)
            
            _y, _X = data[:, 0], data[:, 1:]
            _y = self._sgn(_y)
            
            r = self.r / (1 + t)
            # r = self.r
            self._sgd(X=_X, y=_y, m=m, r=r)
        
            # Compute the total hinge loss for the epoch
            total_loss = self.calc_loss(_X, _y)

            if self.exit_on_converge and np.abs(total_loss - loss[t]) < self.tol:
                if(not self.suppress):
                    print(f"Converged at epoch T={t+1}")
                break
            
            loss[t+1] = total_loss
                    
        return loss[1:]

    def calc_loss(self, X, y):
        hinge_loss = np.mean(np.maximum(0, 1 - y.flatten() * self._predict(X).flatten()))
        regularization = 0.5 * np.sum(self._w**2)
        return self.C * hinge_loss + regularization
    
    def _predict(self, X):
        m, n = X.shape
        # Insert bias term
        X = np.insert(X, 0, np.zeros(shape=m,)+1, axis=1)
        
        z = np.dot(X, self._w)
        return z
    
    def predict(self, X: np.ndarray):
        if(self._w is None):
            return
        
        out = self._predict(X)
        
        return out.flatten() >= 0