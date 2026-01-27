"""
Sampler class for Design of Experiments and parameter space exploration.

This module provides the sampler_cls class that combines DOE strategies
with parameter transformations and QoI computation.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import matplotlib.pyplot as plt

try:
    from distributions import DistributionFactory
    from doe_strategies import DOEFactory
except ImportError:  # When imported as a package
    from .distributions import DistributionFactory
    from .doe_strategies import DOEFactory


class sampler_cls:
    """
    Sampler for parametric studies with multiple distribution types and DOE strategies.

    Attributes:
        types: List of distribution types for each parameter
        params: List of distribution parameters for each parameter
        active_keys: Optional list of parameter names
        compute_QoIs: Callable that computes QoI from parameters
        plot_solution: Optional callable for visualization
        n_out: Number of QoI outputs (auto-detected if None)
    """

    def __init__(
        self,
        types: List[str],
        params: List[list],
        active_keys: Optional[List[str]] = None,
        compute_QoIs: Optional[Callable] = None,
        plot_solution: Optional[Callable] = None,
        n_out: Optional[int] = None,
        seed: Optional[int] = None,
        DOE_type: str = 'LHS',
    ):
        """
        Initialize sampler with distribution specifications.

        Args:
            types: List of distribution types ('uniform', 'log_uniform', 'normal', 'lognormal')
            params: List of [param1, param2] for each dimension
            active_keys: Optional parameter names
            compute_QoIs: Function(params_dict) -> QoI array
            plot_solution: Optional visualization function
            n_out: Number of QoI outputs (auto-detected if None)
            seed: Random seed for reproducible sampling (None for random behavior)
            DOE_type: Design of Experiments type ('PRS', 'LHS', 'QRS', 'SG')

        Raises:
            ValueError: If types and params have mismatched lengths
        """
        if len(types) != len(params):
            raise ValueError(f"Length mismatch: types ({len(types)}) vs params ({len(params)})")

        # Validate all distribution types
        for dist_type in types:
            try:
                DistributionFactory.create(dist_type)
            except ValueError as e:
                raise ValueError(f"Invalid distribution type: {e}")

        self.types = types
        self.params = params
        self.active_keys = active_keys
        self.compute_QoIs = compute_QoIs
        self.seed = int(seed) if seed is not None else None
        self.plot_solution = plot_solution
        self.n_out = n_out
        self.DOE_type = DOE_type

        # Create distribution strategy instances
        self.distributions = [DistributionFactory.create(t) for t in types]

        # Print initialization info
        print("Building your FEA sampler...")
        for i in range(len(self.params)):
            if self.active_keys is not None:
                print(f"parameter dimension {i}: {self.active_keys[i]}")
            print(f"distribution type for dimension {i}: {self.types[i]}")
            print(f"params of distribution for dimension {i}: {self.params[i]}")

        # Auto-detect n_out if not provided
        if n_out is None and compute_QoIs is not None:
            self._detect_n_out()

        if compute_QoIs is not None:
            print(f"n_out: {self.n_out}")

        # Storage for samples (X is computed on-demand from X_normalised)
        self.X_normalised: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.sampler_doe: Optional[object] = None

        print("... done building")

    @property
    def X(self) -> Optional[np.ndarray]:
        """Denormalized physical-space samples (computed from X_normalised on demand)."""
        if self.X_normalised is None:
            return None
        return self._denormalise_samples(self.X_normalised)

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs by calling compute_QoIs at center point."""
        print("n_out not provided -> calling FEA solver for automatic determination")

        X = self._get_center_point()

        if self.active_keys is not None:
            params_dict = {self.active_keys[i]: X[0, i] for i in range(len(self.active_keys))}
            QoIs = self.compute_QoIs(params_dict)
        else:
            QoIs = self.compute_QoIs(X)

        print(f"QoIs at test point: {QoIs}")
        self.n_out = np.max(QoIs.shape)

    def _get_center_point(self) -> np.ndarray:
        """Get center point in physical space based on distribution types."""
        X = np.zeros((1, len(self.params)))

        for i in range(len(self.params)):
            dist_type = self.types[i]
            param = self.params[i]

            if dist_type == 'uniform':
                X[0, i] = (param[0] + param[1]) / 2
            elif dist_type == 'log_uniform':
                X[0, i] = np.exp((param[0] + param[1]) / 2)
            elif dist_type == 'normal':
                X[0, i] = param[0]  # Use mean
            elif dist_type == 'lognormal':
                X[0, i] = np.exp(param[0])  # Use exp(mean)

        return X

    def sampling(
        self,
        N: int,
        DOE_type: str = 'PRS',
        as_additional_points: bool = False,
        plot: bool = False,
        sample_in_batch: bool = False,
    ) -> None:
        """
        Generate and evaluate N samples.

        Args:
            N: Number of samples (or refinement level for SG)
            as_additional_points: If True, add to existing samples; if False, replace (default: False)
            plot: If True, plot solutions during sampling
            sample_in_batch: If True, evaluate all samples at once (experimental)

        Raises:
            ValueError: If DOE_type is invalid
        """
        # Setup DOE strategy
        if self.sampler_doe is None or not as_additional_points:
            if self.sampler_doe is not None and not as_additional_points:
                print(f"Warning: Sampling is being reinitialized. Set as_additional_points=True to continue sampling.")
            try:
                self.sampler_doe = DOEFactory.create(self.DOE_type, len(self.params), seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            X_normalised = self.sampler_doe.sample(N, as_additional_points=False)
        else:
            # Reuse existing DOE sampler and append points
            X_normalised = self.sampler_doe.sample(N, as_additional_points=True)

        # Denormalize samples
        X = self._denormalise_samples(X_normalised)

        # Compute QoIs
        if self.n_out is not None:
            Y = np.zeros((len(X_normalised), self.n_out))
        else:
            Y = None

        sample_type = "additional samples" if as_additional_points and self.X_normalised is not None else "samples"
        print(f'Start computing {len(X_normalised)} {sample_type} in parametric dimension {len(self.params)} using {self.DOE_type}')

        # Evaluate samples
        if sample_in_batch and self.compute_QoIs is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X, plot)

        print("... done sampling")

        # Store results
        self._store_results(X_normalised, X, Y, as_additional_points)

    def _denormalise_samples(self, X_normalised: np.ndarray) -> np.ndarray:
        """Transform samples from [-1,1]^n to physical space."""
        X = np.zeros_like(X_normalised)

        for j in range(len(self.params)):
            X[:, j] = self.distributions[j].denormalise(X_normalised[:, j], self.params[j])

        return X

    def _normalise_samples(self, X: np.ndarray) -> np.ndarray:
        """Transform samples from physical space to [-1,1]^n."""
        X_normalised = np.zeros_like(X)

        for j in range(len(self.params)):
            X_normalised[:, j] = self.distributions[j].normalise(X[:, j], self.params[j])

        return X_normalised

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate all samples at once."""
        if self.compute_QoIs is None:
            return None

        if self.active_keys is not None:
            raise NotImplementedError("Batch evaluation not yet implemented for dimension mapping")

        return self.compute_QoIs(X)

    def _evaluate_sequential(self, X: np.ndarray, plot: bool = False) -> Optional[np.ndarray]:
        """Evaluate samples sequentially with optional visualization."""
        if self.compute_QoIs is None:
            return None

        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None

        for i in range(len(X)):
            if self.active_keys is not None:
                params_dict = {self.active_keys[j]: X[i, j] for j in range(len(self.active_keys))}
                Y[i, :] = self.compute_QoIs(params_dict)
            else:
                Y[i, :] = self.compute_QoIs(X[i, :].reshape(1, -1))

            if plot and self.plot_solution is not None:
                self.plot_solution()

        return Y

    def _store_results(
        self,
        X_normalised: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray],
        as_additional_points: bool,
    ) -> None:
        """Store sampling results, optionally appending to existing data."""
        if self.X_normalised is None or not as_additional_points:
            self.X_normalised = X_normalised
            self.Y = Y
        else:
            self.X_normalised = np.concatenate((self.X_normalised, X_normalised), axis=0)
            if Y is not None and self.Y is not None:
                self.Y = np.concatenate((self.Y, Y), axis=0)

    def normalise(self, X: np.ndarray) -> np.ndarray:
        """
        Transform samples from physical space to [-1,1]^n.

        Args:
            X: Samples in physical space, shape (n_samples, n_dims)

        Returns:
            Normalized samples in [-1,1]^n
        """
        return self._normalise_samples(X)

    def denormalise(self, X_normalised: np.ndarray) -> np.ndarray:
        """
        Transform samples from [-1,1]^n to physical space.

        Args:
            X_normalised: Normalized samples in [-1,1]^n, shape (n_samples, n_dims)

        Returns:
            Samples in physical space
        """
        return self._denormalise_samples(X_normalised)

    def plot_scatter(self, clabel: Optional[str] = None, normalised: bool = False, show: bool = True) -> None:
        """
        Plot 1D or 2D sample distribution with optional QoI coloring.

        Args:
            clabel: Label for colorbar
            normalised: If True, plot in normalized [-1,1]^n space; else physical space
            show: If True, display plot

        Raises:
            ValueError: If problem dimensionality is not 1 or 2
        """
        from util import prediction_plot

        if len(self.params) not in [1, 2]:
            raise ValueError("Can only plot 1D or 2D problems")

        X_plot = self.X_normalised if normalised else self.X

        xlabel = None
        ylabel = None
        if self.active_keys is not None:
            if len(self.active_keys) >= 1:
                xlabel = self.active_keys[0]
            if len(self.active_keys) >= 2:
                ylabel = self.active_keys[1]

        prediction_plot(X=X_plot, y=self.Y, clabel=clabel, xlabel=xlabel, ylabel=ylabel, show=show)
