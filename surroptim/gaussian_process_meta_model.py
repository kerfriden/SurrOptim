"""
Gaussian Process metamodel for regression.

This module provides a Gaussian Process regressor wrapper around sklearn's GaussianProcessRegressor
with RBF kernel and white noise for robust inference.
"""

import numpy as np
from surroptim.meta_models import metamodel

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
except ImportError:
    raise ImportError("sklearn is required for Gaussian Process regression")


class GP_regressor(metamodel):
    """Gaussian Process regressor with RBF kernel and white noise."""

    def __init__(
        self,
        length_scale: float = 1.0,
        length_scale_bounds: tuple = (1.0e-2, 1.0e2),
        noise_level_bounds: tuple = (1e-10, 1e-3),
    ):
        """
        Initialize Gaussian Process regressor.

        Args:
            length_scale: Initial RBF length scale parameter
            length_scale_bounds: Bounds for length scale optimization during training
            noise_level_bounds: Bounds for white noise level optimization during training
        """
        super().__init__()
        
        self.kernel = (
            1.0 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
            + WhiteKernel(noise_level=1, noise_level_bounds=noise_level_bounds)
        )
        self.model = GaussianProcessRegressor(kernel=self.kernel, random_state=10)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gaussian Process model.

        Args:
            X: Training samples, shape (n_samples, n_dims)
            y: Training targets, shape (n_samples,) or (n_samples, n_outputs)
        """
        super().train_init(X, y)
        self.model.fit(X, y)
        print("Optimised kernel parameters:", self.model.kernel_)

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Predict at new points using the trained GP model.

        Args:
            X: Points to predict at, shape (n_samples, n_dims)
            return_std: If True, also return predictive standard deviation

        Returns:
            predictions: Mean predictions, shape (n_samples,) or (n_samples, n_outputs)
            std (optional): Predictive standard deviation if return_std=True
        """
        return self.model.predict(X, return_std=return_std)
