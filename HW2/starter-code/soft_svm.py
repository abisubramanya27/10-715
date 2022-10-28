import numpy as np

class SoftSVM(object):
    def __init__(self, C):
        """
        Soft Support Vector Machine Classifier
        The soft SVM algorithm classifies data points of dimension `d` 
        (this dimension includes the bias) into {-1, +1} classes.
        It receives a regularization parameter `C` that
        controls the margin penalization.
        """
        self.C = C

    def predict(self, X: np.ndarray) ->  np.ndarray:
        """
        Input
        ----------
        X: numpy array of shape (n, d)

        Return
        ------
        y_hat: numpy array of shape (n, )
        """

        # TODO: Make predictions
        y_hat = np.where(X @ self.w >= 0, 1, -1)
        assert y_hat.shape==(len(X),),\
            f'Check your y_hat dimensions they should be {(len(X),)} and are {y_hat.shape}'
        return y_hat

    def subgradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        subgrad: numpy array of shape (d, )
        """

        # TODO: Compute the subgradient
        y = y.reshape((-1,1))
        w = self.w.reshape((-1,1))
        non_zero_grad = y*(X @ w) < 1
        subgrad = self.w - self.C * np.sum(y * non_zero_grad * X, axis=0)
        assert subgrad.shape==(X.shape[1],),\
            f'Check your subgrad dimensions they should be {(X.shape[1],)} and are {subgrad.shape}'
        return subgrad

    def loss(self, X: np.ndarray, y: np.ndarray):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        svm_loss: float
        """

        # TODO: write the soft svm loss that incorporates regularization and hinge loss
        # Using average for the loss, sicne it will be easier to compare train and test loss, and analyze
        hinge_loss = self.C * np.mean(np.maximum(1 - y*(X@self.w), 0))
        reg_loss = 0.5 * np.sum(self.w ** 2)
        svm_loss = hinge_loss + reg_loss
        return svm_loss
    
    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        accuracy: float
        """

        # TODO: Evaluate the accuracy of the model on the dataset
        y_hat = self.predict(X)
        accuracy = 100. * np.mean(y_hat == y)
        return accuracy

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              n_iterations: int, learning_rate: float,
              random_seed=1) -> None:
        """
        Input
        ----------
        X_train: numpy array of shape (n, d)
        y_train: numpy array of shape (n, )
        n_iterations: int
        learning_rate: float
        random_seed: int
        """
        
        # Check inputs
        assert len(X_train)==len(y_train)
        assert np.array_equal(np.sort(np.unique(y_train)), np.array([-1, 1]))
        
        # Initialize model
        np.random.seed(random_seed)
        self.d = X_train.shape[1]
        self.w = np.random.normal(size=(self.d,))

        for t in range(n_iterations):
            # TODO: Update weights according to training procedure
            self.w -= learning_rate*self.subgradient(X_train, y_train)
