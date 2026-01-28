"""
Sampler class for Design of Experiments and parameter space exploration.

This module provides the sampler_cls class that combines DOE strategies
with parameter transformations and QoI computation.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import matplotlib.pyplot as plt

from .distributions import DistributionFactory
from .doe_strategies import DOEFactory


class sampler_cls:
    """
    Sampler for parametric studies with multiple distribution types and DOE strategies.

    Attributes:
        distributions: List of distribution types for each parameter
        bounds: List of distribution parameters for each parameter
        active_keys: Optional list of parameter names
        compute_QoIs: Callable that computes QoI from parameters
        plot_solution: Optional callable for visualization
        n_out: Number of QoI outputs (auto-detected if None)
    """

    def __init__(
        self,
        distributions: List[str],
        bounds: List[list],
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
            distributions: List of distribution types ('uniform', 'log_uniform', 'normal', 'lognormal')
            bounds: List of [param1, param2] for each dimension
            active_keys: Optional parameter names
            compute_QoIs: Function(params_dict) -> QoI array
            plot_solution: Optional visualization function
            n_out: Number of QoI outputs (auto-detected if None)
            seed: Random seed for reproducible sampling (None for random behavior)
            DOE_type: Design of Experiments type ('PRS', 'LHS', 'QRS', 'SG')

        Raises:
            ValueError: If distributions and bounds have mismatched lengths
        """
        if len(distributions) != len(bounds):
            raise ValueError(f"Length mismatch: distributions ({len(distributions)}) vs bounds ({len(bounds)})")

        # Validate all distribution types
        for dist_type in distributions:
            try:
                DistributionFactory.create(dist_type)
            except ValueError as e:
                raise ValueError(f"Invalid distribution type: {e}")

        self.distributions = distributions
        self.bounds = bounds
        self.active_keys = active_keys
        self.compute_QoIs = compute_QoIs
        self.seed = int(seed) if seed is not None else None
        self.plot_solution = plot_solution
        self.n_out = n_out
        self.DOE_type = DOE_type

        # Create distribution strategy instances
        self.distribution_strategies = [DistributionFactory.create(t) for t in distributions]

        # Print initialization info
        print("Building your FEA sampler...")
        for i in range(len(self.bounds)):
            if self.active_keys is not None:
                print(f"parameter dimension {i}: {self.active_keys[i]}")
            print(f"distribution type for dimension {i}: {self.distributions[i]}")
            print(f"bounds of distribution for dimension {i}: {self.bounds[i]}")

        # Auto-detect n_out if not provided
        if n_out is None and compute_QoIs is not None:
            self._detect_n_out()

        if compute_QoIs is not None:
            print(f"n_out: {self.n_out}")

        # Storage for samples (X is computed on-demand from X_reference)
        self.X_reference: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.sampler_doe: Optional[object] = None

        print("... done building")

    @property
    def X(self) -> Optional[np.ndarray]:
        """Denormalized physical-space samples (computed from X_reference on demand)."""
        if self.X_reference is None:
            return None
        return self._reference_to_physical_samples(self.X_reference)

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
        X = np.zeros((1, len(self.bounds)))

        for i in range(len(self.bounds)):
            dist_type = self.distributions[i]
            param = self.bounds[i]

            if dist_type == 'uniform':
                X[0, i] = (param[0] + param[1]) / 2
            elif dist_type == 'log_uniform':
                # Geometric mean when bounds are provided in physical space
                if param[0] <= 0 or param[1] <= 0:
                    raise ValueError("Log-uniform bounds must be positive for center point computation")
                X[0, i] = np.exp((np.log(param[0]) + np.log(param[1])) / 2)
            elif dist_type == 'normal':
                X[0, i] = param[0]  # Use mean
            elif dist_type == 'lognormal':
                X[0, i] = np.exp(param[0])  # Use exp(mean)

        return X

    def sample(
        self,
        N: int,
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
                print(
                    "Warning: Sampling is being reinitialized. Set as_additional_points=True to continue sampling."
                )
            try:
                self.sampler_doe = DOEFactory.create(self.DOE_type, len(self.bounds), seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            X_reference = self.sampler_doe.sample(N, as_additional_points=False)
        else:
            # Reuse existing DOE sampler and append points
            X_reference = self.sampler_doe.sample(N, as_additional_points=True)

        # Denormalize samples
        X = self._reference_to_physical_samples(X_reference)

        # Compute QoIs
        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        sample_type = "additional samples" if as_additional_points and self.X_reference is not None else "samples"
        print(
            f"Start computing {len(X_reference)} {sample_type} in parametric dimension {len(self.bounds)} using {self.DOE_type}"
        )

        # Evaluate samples
        if sample_in_batch and self.compute_QoIs is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X, plot)

        print("... done sampling")

        # Store results
        self._store_results(X_reference, X, Y, as_additional_points)

    def _reference_to_physical_samples(self, X_reference: np.ndarray) -> np.ndarray:
        """Transform samples from reference space [-1,1]^n to physical space."""
        X = np.zeros_like(X_reference)

        for j in range(len(self.bounds)):
            X[:, j] = self.distribution_strategies[j].denormalise(X_reference[:, j], self.bounds[j])

        return X

    def _physical_to_reference_samples(self, X: np.ndarray) -> np.ndarray:
        """Transform samples from physical space to reference space [-1,1]^n."""
        X_reference = np.zeros_like(X)

        for j in range(len(self.bounds)):
            X_reference[:, j] = self.distribution_strategies[j].normalise(X[:, j], self.bounds[j])

        return X_reference

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate all samples at once."""
        if self.compute_QoIs is None:
            return None

        if self.active_keys is not None:
            raise NotImplementedError("Batch evaluation not yet implemented for dimension mapping")

        try:
            return self.compute_QoIs(X)
        except TypeError as e:
            raise TypeError(
                f"QoI evaluation failed: {e}. "
                f"Expected array input, but the QoI might be expecting a dict (active_keys={self.active_keys}). "
                f"Ensure QoI signature matches the input type."
            ) from e

    def _evaluate_sequential(self, X: np.ndarray, plot: bool = False) -> Optional[np.ndarray]:
        """Evaluate samples sequentially with optional visualization."""
        if self.compute_QoIs is None:
            return None

        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None

        for i in range(len(X)):
            try:
                if self.active_keys is not None:
                    params_dict = {self.active_keys[j]: X[i, j] for j in range(len(self.active_keys))}
                    Y[i, :] = self.compute_QoIs(params_dict)
                else:
                    Y[i, :] = self.compute_QoIs(X[i, :].reshape(1, -1))
            except TypeError as e:
                if self.active_keys is not None:
                    raise TypeError(
                        f"QoI evaluation failed at sample {i}: {e}. "
                        f"Expected dict input (active_keys={self.active_keys}), "
                        f"but QoI might be expecting array. "
                        f"Ensure QoI signature accepts: {{{', '.join(repr(k) for k in self.active_keys)}: float}}"
                    ) from e
                else:
                    raise TypeError(
                        f"QoI evaluation failed at sample {i}: {e}. "
                        f"Expected array input (active_keys=None), "
                        f"but QoI might be expecting dict. "
                        f"Pass active_keys parameter if QoI expects named parameters."
                    ) from e

            if plot and self.plot_solution is not None:
                self.plot_solution()

        return Y

    def _store_results(
        self,
        X_reference: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray],
        as_additional_points: bool,
    ) -> None:
        """Store sampling results, optionally appending to existing data."""
        if self.X_reference is None or not as_additional_points:
            self.X_reference = X_reference
            self.Y = Y
        else:
            self.X_reference = np.concatenate((self.X_reference, X_reference), axis=0)
            if Y is not None and self.Y is not None:
                self.Y = np.concatenate((self.Y, Y), axis=0)

    def physical_to_reference(self, X: np.ndarray) -> np.ndarray:
        """
        Transform samples from physical space to reference space [-1,1]^n.

        Args:
            X: Samples in physical space, shape (n_samples, n_dims)

        Returns:
            Reference samples in [-1,1]^n
        """
        return self._physical_to_reference_samples(X)

    def reference_to_physical(self, X_reference: np.ndarray) -> np.ndarray:
        """
        Transform samples from reference space [-1,1]^n to physical space.

        Args:
            X_reference: Reference samples in [-1,1]^n, shape (n_samples, n_dims)

        Returns:
            Samples in physical space
        """
        return self._reference_to_physical_samples(X_reference)

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
        from .util import prediction_plot

        if len(self.bounds) not in [1, 2]:
            raise ValueError("Can only plot 1D or 2D problems")

        X_plot = self.X_reference if normalised else self.X

        xlabel = None
        ylabel = None
        if self.active_keys is not None:
            if len(self.active_keys) >= 1:
                xlabel = self.active_keys[0]
            if len(self.active_keys) >= 2:
                ylabel = self.active_keys[1]

        prediction_plot(X=X_plot, y=self.Y, clabel=clabel, xlabel=xlabel, ylabel=ylabel, show=show)


class sampler_new_cls:
    """Sampler that uses an external `params` processor for unit<->physical transforms.

    The `params` argument must implement the same API as `params_cls` (e.g. `reference_to_physical`,
    `physical_to_reference`, `pack`, `unpack`, and `dim`). This sampler delegates normalization
    logic to the provided `params` instance instead of keeping its own distribution strategies.
    """

    def __init__(
        self,
        params,
        seed: Optional[int] = None,
        DOE_type: str = 'LHS',
    ):
        # params is expected to encapsulate parameter layout and optionally
        # metadata like compute_QoIs, active_keys, and n_out.
        self.params = params
        self.compute_QoIs = getattr(params, "compute_QoIs", None)
        self.active_keys = getattr(params, "active_keys", None)
        self.seed = int(seed) if seed is not None else None
        # plotting is not supported by sampler_new_cls; omit plot_solution
        # n_out is not stored on params; detect from compute_QoIs when needed
        self.n_out = None
        self.DOE_type = DOE_type

        # Use the parameter processor's dimension
        self.dim = int(self.params.dim)

        # DOE sampler placeholder
        self.sampler_doe: Optional[object] = None

        print("Building sampler_new_cls using external params processor...")
        print(f"param processor dim: {self.dim}")
        if self.active_keys is not None:
            for i, k in enumerate(self.active_keys):
                print(f"parameter dimension {i}: {k}")

        if self.compute_QoIs is not None:
            self._detect_n_out()
            print(f"n_out: {self.n_out}")

        # Storage for samples
        self.X_reference: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None

        print("... done building sampler_new_cls")

    @property
    def X(self) -> Optional[np.ndarray]:
        if self.X_reference is None:
            return None
        return self.reference_to_physical(self.X_reference)

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs using the params base point."""
        print("n_out not provided -> calling QoI for automatic determination")
        x_center = self.params.pack(self.params.base)
        if self.active_keys is not None:
            params_dict = self.params.unpack(x_center)
            QoIs = self.compute_QoIs(params_dict)
        else:
            QoIs = self.compute_QoIs(x_center)
        print(f"QoIs at test point: {QoIs}")
        self.n_out = np.max(np.asarray(QoIs).shape)

    def sample(self, N: int, as_additional_points: bool = False, plot: bool = False, sample_in_batch: bool = False) -> None:
        if self.sampler_doe is None or not as_additional_points:
            if self.sampler_doe is not None and not as_additional_points:
                print("Warning: Sampling is being reinitialized. Set as_additional_points=True to continue sampling.")
            try:
                self.sampler_doe = DOEFactory.create(self.DOE_type, self.dim, seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            X_reference = self.sampler_doe.sample(N, as_additional_points=False)
        else:
            X_reference = self.sampler_doe.sample(N, as_additional_points=True)

        # Convert reference -> physical using params processor
        X = self.reference_to_physical(X_reference)

        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        sample_type = "additional samples" if as_additional_points and self.X_reference is not None else "samples"
        print(f"Start computing {len(X_reference)} {sample_type} in parametric dimension {self.dim} using {self.DOE_type}")

        if sample_in_batch and self.compute_QoIs is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X, plot)

        print("... done sampling")

        self._store_results(X_reference, X, Y, as_additional_points)

    def reference_to_physical(self, X_reference: np.ndarray) -> np.ndarray:
        """Use params.processor to convert reference to physical."""
        return self.params.reference_to_physical(X_reference)

    def physical_to_reference(self, X: np.ndarray) -> np.ndarray:
        """Use params.processor to convert physical to reference."""
        return self.params.physical_to_reference(X)

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        if self.compute_QoIs is None:
            return None
        if self.active_keys is not None:
            raise NotImplementedError("Batch evaluation not implemented for active_keys mapping")
        try:
            return self.compute_QoIs(X)
        except TypeError as e:
            raise TypeError(f"QoI evaluation failed: {e}") from e

    def _evaluate_sequential(self, X: np.ndarray, plot: bool = False) -> Optional[np.ndarray]:
        if self.compute_QoIs is None:
            return None
        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None
        for i in range(len(X)):
            try:
                if self.active_keys is not None:
                    params_dict = self.params.unpack(X[i]) if hasattr(self.params, 'unpack') else {self.active_keys[j]: X[i,j] for j in range(len(self.active_keys))}
                    Y[i, :] = self.compute_QoIs(params_dict)
                else:
                    Y[i, :] = self.compute_QoIs(X[i, :].reshape(1, -1))
            except TypeError as e:
                raise
            # plotting not supported in sampler_new_cls
        return Y

    def _store_results(self, X_reference: np.ndarray, X: np.ndarray, Y: Optional[np.ndarray], as_additional_points: bool) -> None:
        if self.X_reference is None or not as_additional_points:
            self.X_reference = X_reference
            self.Y = Y
        else:
            self.X_reference = np.concatenate((self.X_reference, X_reference), axis=0)
            if Y is not None and self.Y is not None:
                self.Y = np.concatenate((self.Y, Y), axis=0)
    
