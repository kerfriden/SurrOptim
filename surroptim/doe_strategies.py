"""
DOE (Design of Experiments) strategy classes.

This module provides abstract base class and concrete implementations
for different sampling methods (PRS, LHS, QRS, Sparse Grid).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np

try:
    from config import QRS_SOBOL_M, SUPPORTED_DOE_TYPES
except ImportError:  # When imported as a package
    from .config import QRS_SOBOL_M, SUPPORTED_DOE_TYPES


class DOEStrategy(ABC):
    """Abstract base class for Design of Experiments strategies."""

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """
        Generate N samples in [-1,1]^dim.

        Args:
            N: Number of samples to generate
            as_additional_points: If True, add to existing samples; if False, replace

        Returns:
            numpy array of shape (N, dim) with samples in [-1,1]
        """
        pass


class PRSStrategy(DOEStrategy):
    """Pseudo-Random Sampling (uniform random)."""

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.X = None

    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """Generate n_samples uniform random samples in [-1,1]^dim.

        Accepts legacy argument names `N` and `as_additional_points` for backward compatibility.
        """
        # resolve legacy names
        if n_samples is None:
            if N is None:
                raise ValueError("n_samples (or legacy N) must be provided")
            n_samples = N
        if as_additional_points is not None:
            as_additional_samples = as_additional_points

        if n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {n_samples}")

        X = 2 * self.rng.random((n_samples, self.dim)) - 1

        if as_additional_samples and self.X is not None:
            self.X = np.concatenate((self.X, X), axis=0)
        else:
            self.X = X

        return X


class LHSStrategy(DOEStrategy):
    """Latin Hypercube Sampling."""

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.dim = dim
        self.seed = seed
        self.X = None

    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """Generate n_samples Latin Hypercube samples in [-1,1]^dim.

        Accepts legacy argument names `N` and `as_additional_points` for backward compatibility.
        """
        # resolve legacy names
        if n_samples is None:
            if N is None:
                raise ValueError("n_samples (or legacy N) must be provided")
            n_samples = N
        if as_additional_points is not None:
            as_additional_samples = as_additional_points

        if n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {n_samples}")

        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
        X = 2 * sampler.random(n=n_samples) - 1

        if as_additional_samples and self.X is not None:
            self.X = np.concatenate((self.X, X), axis=0)
        else:
            self.X = X

        return X


class QRSStrategy(DOEStrategy):
    """Quasi-Random Sampling using Sobol sequence."""

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.dim = dim
        self.seed = seed
        from scipy.stats import qmc
        # Sobol uses scramble parameter with seed for randomization
        scramble = seed is not None
        self.sampler = qmc.Sobol(d=self.dim, scramble=scramble, seed=seed)
        self.X_all = self.sampler.random_base2(m=QRS_SOBOL_M)
        self.X = None
        self.index = 0

    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """Draw n_samples points from Sobol sequence in [-1,1]^dim.

        Accepts legacy argument names `N` and `as_additional_points` for backward compatibility.
        """
        # resolve legacy names
        if n_samples is None:
            if N is None:
                raise ValueError("n_samples (or legacy N) must be provided")
            n_samples = N
        if as_additional_points is not None:
            as_additional_samples = as_additional_points

        if n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {n_samples}")
        if n_samples + self.index > len(self.X_all):
            raise ValueError(
                f"Requested {n_samples} samples at index {self.index}, but only "
                f"{len(self.X_all)} points available"
            )

        if as_additional_samples and self.X is not None:
            X = 2 * self.X_all[self.index:(self.index + n_samples), :] - 1
            self.X = 2 * self.X_all[:(self.index + n_samples), :] - 1
        else:
            X = 2 * self.X_all[:n_samples, :] - 1
            self.X = X

        self.index += n_samples
        return X


class SGStrategy(DOEStrategy):
    """Sparse Grid sampling using Clenshaw-Curtis quadrature."""

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.dim = dim
        self.seed = seed  # Sparse grids are deterministic, seed not used
        self.X = None

    def sample(self, n_samples: Optional[int] = None, N: Optional[int] = None, as_additional_samples: bool = False, as_additional_points: Optional[bool] = None) -> np.ndarray:
        """Generate sparse grid samples (n_samples is refinement level).

        Accepts legacy argument names `N` and `as_additional_points` for backward compatibility.
        """
        # resolve legacy names
        if n_samples is None:
            if N is None:
                raise ValueError("n_samples (or legacy N) must be provided")
            n_samples = N
        if as_additional_points is not None:
            as_additional_samples = as_additional_points

        if n_samples <= 0:
            raise ValueError(f"Refinement level must be positive, got {n_samples}")

        from surroptim.sparse_grid import generate_sparse_grid

        # n_samples represents the refinement level for sparse grids; already in [-1,1]
        X = generate_sparse_grid(self.dim, n_samples)

        if as_additional_samples and self.X is not None:
            # Find points in X not already in self.X
            mask = np.isin(
                X.view([('', X.dtype)] * X.shape[1]),
                self.X.view([('', self.X.dtype)] * self.X.shape[1])
            )
            X_new = X[~mask.any(axis=1)]
            self.X = np.concatenate((self.X, X_new), axis=0)
            return X_new
        else:
            self.X = X
            return X


class DOEFactory:
    """Factory for creating DOE strategy instances."""

    _strategies = {
        'PRS': PRSStrategy,
        'LHS': LHSStrategy,
        'QRS': QRSStrategy,
        'SG': SGStrategy,
    }

    @classmethod
    def create(cls, doe_type: str, dim: int, seed: Optional[int] = None) -> DOEStrategy:
        """
        Create a DOE strategy instance.

        Args:
            doe_type: Type of DOE ('PRS', 'LHS', 'QRS', 'SG')
            dim: Problem dimensionality
            seed: Random seed for reproducible sampling

        Returns:
            DOEStrategy instance

        Raises:
            ValueError: If DOE type is not supported or parameters invalid
        """
        if doe_type not in cls._strategies:
            raise ValueError(
                f"Unknown DOE type: '{doe_type}'. "
                f"Supported types: {list(cls._strategies.keys())}"
            )
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")

        return cls._strategies[doe_type](dim, seed=seed)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Return list of supported DOE types."""
        return list(cls._strategies.keys())
