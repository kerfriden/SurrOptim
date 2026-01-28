"""
DOE (Design of Experiments) class for backwards compatibility.

This module provides the DOE_cls class that wraps the new DOEFactory
for backward compatibility with existing code.
"""

import numpy as np
from typing import Optional

try:
    from doe_strategies import DOEFactory
except ImportError:  # When imported as a package
    from .doe_strategies import DOEFactory


class DOE_cls:
    """
    Legacy DOE class for backward compatibility.
    
    Wraps the new DOEFactory-based strategy pattern. New code should
    use DOEFactory directly or through sampler_cls.
    """

    def __init__(self, dim: int, doe_type: str = "PRS", DOE_type: Optional[str] = None):
        """
        Initialize DOE sampler.

        Args:
            dim: Problem dimensionality
            DOE_type: Sampling strategy ('PRS', 'LHS', 'QRS', 'SG')
        """
        # accept legacy DOE_type kwarg
        actual = DOE_type if DOE_type is not None else doe_type
        print(f"Instantiating {actual} sampler")
        self.dim = dim
        self.doe_type = actual
        self.strategy = DOEFactory.create(actual, dim)
        self.X = None  # Store accumulated samples when using as_additional_samples

    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """
        Generate N samples in [-1,1]^dim.

        Args:
            N: Number of samples
            as_additional_points: If True, add to existing samples

        Returns:
            numpy array of shape (N, dim) with samples in [-1,1]
        """
        # resolve legacy kwargs
        if n_samples is None:
            if N is None:
                raise ValueError("n_samples (or legacy N) must be provided")
            n_samples = N
        if as_additional_points is not None:
            as_additional_samples = as_additional_points

        X_new = self.strategy.sample(n_samples, as_additional_samples=as_additional_samples)

        if as_additional_samples:
            if self.X is None:
                self.X = X_new
            else:
                self.X = np.vstack([self.X, X_new])
        else:
            self.X = X_new
        
        return X_new
