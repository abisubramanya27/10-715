import numpy as np

class Perceptron(object):
    def __init__(self, d):
        """
        Perceptron Classifier
        The perceptron algorithm classifies data points of dimensionality `d`
        into {-1, +1} classes.
        """
        self.d = d
        self.w = np.zeros(d)  # Don't change this
        self.b = 0.0  # Don't change this

    def predict(self, x:  np.ndarray) -> np.ndarray:
        # TODO: Complete the predict method - Done
        y_hat = x @ self.w + self.b
        assert y_hat.shape == (x.shape[0],)
        y_hat = np.where(y_hat > 0, 1, -1)
        return y_hat

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        # TODO: Complete the update method - Done
        self.w += x * y
        self.b += y
        assert self.w.shape == (self.d,),\
            f'Check your weight dimensions. Expected: {(self.d,)}. Actual: {self.w.shape}.'

    def _exists_misclassified_sample(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.int64:
        return np.sum(y_hat != y_true) > 0
    
    def _calc_accuracies(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.float64:
        return (np.sum(y_hat == y_true) / np.prod(y_true.shape)) * 100
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, iterations: int) -> None:       
        t = 0
        # TODO: Write the algorithm and store the trajectories - Done
        y_train_hat = self.predict(X_train)
        y_test_hat = self.predict(X_test)
        self.trajectories = {
            'train': [self._calc_accuracies(y_train_hat, y_train)], 
            'test': [self._calc_accuracies(y_test_hat, y_test)]
        }
        while (self._exists_misclassified_sample(y_train_hat, y_train)) & (t < iterations):
            misclass_sample_idx = np.asarray(y_train_hat != y_train).nonzero()[0][0]
            self.update(X_train[misclass_sample_idx], y_train[misclass_sample_idx])
            y_train_hat, y_test_hat = self.predict(X_train), self.predict(X_test)
            self.trajectories['train'].append(self._calc_accuracies(y_train_hat, y_train))
            self.trajectories['test'].append(self._calc_accuracies(y_test_hat, y_test))
            t += 1
