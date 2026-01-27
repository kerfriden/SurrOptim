"""
Distribution strategy classes for parameter transformations.

This module provides an abstract base class and concrete implementations
for different probability distributions used in sampling.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np
from scipy.stats import norm
try:
    from .config import SUPPORTED_DISTRIBUTIONS
except ImportError:
    from config import SUPPORTED_DISTRIBUTIONS


class DistributionStrategy(ABC):
    """Abstract base class for distribution transformations."""

    @abstractmethod
    def denormalise(self, X_normalised: np.ndarray, params: list) -> np.ndarray:
        """Transform from [-1,1] to physical space."""
        pass

    @abstractmethod
    def normalise(self, X: np.ndarray, params: list) -> np.ndarray:
        """Transform from physical to [-1,1] space."""
        pass


class UniformDistribution(DistributionStrategy):
    """Uniform distribution on [a, b]."""

    def denormalise(self, X_normalised: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from [-1,1] to [a,b]: X_phys = (a+b)/2 + (b-a)/2 * X_norm"""
        if len(params) != 2:
            raise ValueError(f"Uniform distribution requires 2 parameters, got {len(params)}")
        a, b = params[0], params[1]
        if a >= b:
            raise ValueError(f"Invalid uniform bounds: a={a} >= b={b}")
        return (a + b) / 2 + (b - a) / 2 * X_normalised

    def normalise(self, X: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from [a,b] to [-1,1]: X_norm = 2*(X_phys-a)/(b-a) - 1"""
        if len(params) != 2:
            raise ValueError(f"Uniform distribution requires 2 parameters, got {len(params)}")
        a, b = params[0], params[1]
        if a >= b:
            raise ValueError(f"Invalid uniform bounds: a={a} >= b={b}")
        return 2 * (X - a) / (b - a) - 1


class LogUniformDistribution(DistributionStrategy):
    """Log-uniform distribution: uniform on log scale."""

    def denormalise(self, X_normalised: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from [-1,1] to exp space: X_phys = exp((a+b)/2 + (b-a)/2 * X_norm)"""
        if len(params) != 2:
            raise ValueError(f"Log-uniform distribution requires 2 parameters, got {len(params)}")
        a, b = params[0], params[1]
        if a >= b:
            raise ValueError(f"Invalid log-uniform bounds: a={a} >= b={b}")
        return np.exp((a + b) / 2 + (b - a) / 2 * X_normalised)

    def normalise(self, X: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from exp space to [-1,1]: X_norm = 2*(log(X_phys)-a)/(b-a) - 1"""
        if len(params) != 2:
            raise ValueError(f"Log-uniform distribution requires 2 parameters, got {len(params)}")
        a, b = params[0], params[1]
        if a >= b:
            raise ValueError(f"Invalid log-uniform bounds: a={a} >= b={b}")
        if np.any(X <= 0):
            raise ValueError("Log-uniform requires positive samples")
        return 2 * (np.log(X) - a) / (b - a) - 1


class DistributionFactory:
    """Factory for creating distribution strategy instances."""

    _distributions = {
        'uniform': UniformDistribution,
        'log_uniform': LogUniformDistribution,
    }

    @classmethod
    def create(cls, distribution_type: str) -> DistributionStrategy:
        """
        Create a distribution strategy instance.

        Args:
            distribution_type: Type of distribution ('uniform', 'log_uniform', 'normal', 'lognormal')

        Returns:
            DistributionStrategy instance

        Raises:
            ValueError: If distribution type is not supported
        """
        if distribution_type not in cls._distributions:
            raise ValueError(
                f"Unknown distribution type: '{distribution_type}'. "
                f"Supported types: {list(cls._distributions.keys())}"
            )
        return cls._distributions[distribution_type]()

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Return list of supported distribution types."""
        return list(cls._distributions.keys())
