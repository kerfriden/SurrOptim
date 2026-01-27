"""
Distribution strategy classes for parameter transformations.

This module provides an abstract base class and concrete implementations
for different probability distributions used in sampling.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np
from scipy.stats import norm

EPS_POSITIVE = 1e-12  # guard for log-uniform positivity checks
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
    """Log-uniform distribution using specified transform formulas.
    
    Maps reference space u ∈ [-1, 1] to physical space mu ∈ [mu_min, mu_max]
    using the transforms:
        u_to_mu: mu = mu_min * (mu_max/mu_min)^((u+1)/2)
        mu_to_u: u = 2*ln(mu/mu_min)/ln(mu_max/mu_min) - 1
    """

    def denormalise(self, X_normalised: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from [-1,1] to physical space: u -> mu
        
        Args:
            X_normalised: Reference space values in [-1, 1]
            params: [mu_min, mu_max] physical bounds (must be positive)
            
        Returns:
            Physical space values
        """
        if len(params) != 2:
            raise ValueError(f"Log-uniform distribution requires 2 parameters, got {len(params)}")
        mu_min, mu_max = params[0], params[1]
        if not np.isfinite(mu_min) or not np.isfinite(mu_max):
            raise ValueError(f"Log-uniform bounds must be finite, got mu_min={mu_min}, mu_max={mu_max}")
        if mu_min <= 0 or mu_max <= 0:
            raise ValueError(f"Log-uniform bounds must be positive, got mu_min={mu_min}, mu_max={mu_max}")
        if mu_min >= mu_max:
            raise ValueError(f"Invalid log-uniform bounds: mu_min={mu_min} >= mu_max={mu_max}")

        # u_to_mu: mu = mu_min * (mu_max/mu_min)^((u+1)/2)
        return mu_min * (mu_max / mu_min) ** ((X_normalised + 1) / 2)

    def normalise(self, X: np.ndarray, params: List[float]) -> np.ndarray:
        """Transform from physical space to [-1,1]: mu -> u
        
        Args:
            X: Physical space values
            params: [mu_min, mu_max] physical bounds (must be positive)
            
        Returns:
            Reference space values in [-1, 1]
        """
        if len(params) != 2:
            raise ValueError(f"Log-uniform distribution requires 2 parameters, got {len(params)}")
        mu_min, mu_max = params[0], params[1]
        if not np.isfinite(mu_min) or not np.isfinite(mu_max):
            raise ValueError(f"Log-uniform bounds must be finite, got mu_min={mu_min}, mu_max={mu_max}")
        if mu_min <= 0 or mu_max <= 0:
            raise ValueError(f"Log-uniform bounds must be positive, got mu_min={mu_min}, mu_max={mu_max}")
        if mu_min >= mu_max:
            raise ValueError(f"Invalid log-uniform bounds: mu_min={mu_min} >= mu_max={mu_max}")

        X_arr = np.asarray(X)
        if np.any(X_arr <= 0):
            raise ValueError("Log-uniform requires positive samples")

        # mu_to_u: u = 2*ln(mu/mu_min)/ln(mu_max/mu_min) - 1
        return 2 * (np.log(X_arr) - np.log(mu_min)) / (np.log(mu_max) - np.log(mu_min)) - 1


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
