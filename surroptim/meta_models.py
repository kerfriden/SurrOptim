import numpy as np
from surroptim.util import r2_score


class metamodel:
    """Base metamodel class with train/predict/test interface."""

    def __init__(self) -> None:
        self.dim = None
        self.n_out = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None

    def train_init(self, X: np.ndarray, y: np.ndarray) -> None:
        if X is None:
            raise ValueError("No training data X provided")
        if y is None:
            raise ValueError("No training data y provided")
        X = np.asarray(X)
        y = np.asarray(y)

        # Normalize shapes:
        # - X is always treated as (N, dim)
        # - y is always treated as (N, n_out)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 0:
            y = y.reshape(1, 1)
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            # Treat the first axis as samples and flatten the remaining axes
            y = y.reshape(y.shape[0], -1)

        # Handle common accidental transpose: (n_out, N) instead of (N, n_out)
        if X.ndim == 2 and y.ndim == 2 and y.shape[0] != X.shape[0] and y.shape[1] == X.shape[0]:
            y = y.T

        self.X = X
        self.y = y
        self.dim = X.shape[1]
        self.n_out = y.shape[1]

    def train(self) -> None:
        raise NotImplementedError("train must be implemented in subclasses")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict must be implemented in subclasses")

    def test(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        if self.X is None or self.y is None:
            raise ValueError("Training not done so far")
        if X_test is None:
            raise ValueError("No test dataset X_test provided")
        if y_test is None:
            raise ValueError("No test dataset y_test provided")

        self.X_test = X_test
        self.y_test = y_test

        y_train_pred = self.predict(self.X)
        print("R2 score train set :", r2_score(self.y, y_train_pred))

        y_test_pred = self.predict(self.X_test)
        print("R2 score test set :", r2_score(self.y_test, y_test_pred))
