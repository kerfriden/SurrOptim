"""
k-Nearest Neighbors metamodel for regression.

This module provides a k-NN regressor wrapper around sklearn's KNeighborsRegressor.
"""

import numpy as np
from surroptim.meta_models import metamodel

try:
    from sklearn.neighbors import KNeighborsRegressor
except ImportError:
    raise ImportError("sklearn is required for k-NN regression")


class NNeigh_regressor(metamodel):
    """k-Nearest Neighbors regressor for non-parametric regression."""

    def __init__(self, n_neighbors: int = 1, weights: str = "distance"):
        """
        Initialize k-NN regressor.

        Args:
            n_neighbors: Number of neighbors to use for prediction
            weights: Weight function used in prediction. Options:
                - 'uniform': all points equally weighted
                - 'distance': points weighted by inverse distance
                See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        print(f"k-NN regressor: n_neighbors={self.n_neighbors}, weights='{weights}'")
        
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=weights)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the k-NN model.

        Args:
            X: Training samples, shape (n_samples, n_dims)
            y: Training targets, shape (n_samples,) or (n_samples, n_outputs)
        """
        super().train_init(X, y)
        self.model.fit(X, y)
        print(f"k-NN model trained with {len(X)} samples")

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Predict on new samples.

        Args:
            X: Input samples, shape (n_samples, n_dims)
            return_std: Ignored (k-NN does not provide uncertainty estimates)

        Returns:
            Predictions, shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if return_std:
            # k-NN does not provide uncertainty estimates; just return predictions
            return self.model.predict(X), None
        
        return self.model.predict(X)
