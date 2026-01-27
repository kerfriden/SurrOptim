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

    def __init__(self, dim: int, DOE_type: str = "PRS"):
        """
        Initialize DOE sampler.

        Args:
            dim: Problem dimensionality
            DOE_type: Sampling strategy ('PRS', 'LHS', 'QRS', 'SG')
        """
        print(f"Instantiating {DOE_type} sampler")
        self.dim = dim
        self.DOE_type = DOE_type
        self.strategy = DOEFactory.create(DOE_type, dim)
        self.X = None  # Store accumulated samples when using as_additional_points

    def sample(self, N: int, as_additional_points: bool = False) -> np.ndarray:
        """
        Generate N samples in [-1,1]^dim.

        Args:
            N: Number of samples
            as_additional_points: If True, add to existing samples

        Returns:
            numpy array of shape (N, dim) with samples in [-1,1]
        """
        X_new = self.strategy.sample(N, as_additional_points=as_additional_points)
        
        if as_additional_points:
            if self.X is None:
                self.X = X_new
            else:
                self.X = np.vstack([self.X, X_new])
        else:
            self.X = X_new
        
        return X_new
