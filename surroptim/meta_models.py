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
        self.r2_train = None
        self.r2_test = None

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

    def test(self, X_test: np.ndarray = None, y_test: np.ndarray = None):
        if self.X is None or self.y is None:
            raise ValueError("Training not done so far")

        # Always compute and report train R2
        y_train_pred = self.predict(self.X)
        r2_train = r2_score(self.y, y_train_pred)
        self.r2_train = r2_train
        print("R2 score train set :", r2_train)

        # If no test data provided, only return train R2.
        if X_test is None and y_test is None:
            return r2_train

        # If only one is provided, that's an error.
        if X_test is None:
            raise ValueError("No test dataset X_test provided")
        if y_test is None:
            raise ValueError("No test dataset y_test provided")

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Normalize y_test to (N, n_out) where possible.
        if y_test.ndim == 0:
            y_test = y_test.reshape(1, 1)
        elif y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        elif y_test.ndim > 2:
            y_test = y_test.reshape(y_test.shape[0], -1)
        else:
            # Heuristic: if caller provided a single-sample tensor without the
            # sample axis (e.g. (2,2)) and X_test has one sample, interpret it
            # as one sample with multiple outputs.
            if X_test.ndim == 2 and X_test.shape[0] == 1 and y_test.shape[0] != 1:
                y_test = y_test.reshape(1, -1)

        self.X_test = X_test
        self.y_test = y_test

        y_test_pred = self.predict(self.X_test)
        r2_test = r2_score(self.y_test, y_test_pred)
        self.r2_test = r2_test
        print("R2 score test set :", r2_test)

        return (r2_train, r2_test)
