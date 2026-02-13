"""
Sampler class for Design of Experiments and parameter space exploration.

This module provides the sampler_cls class that combines DOE strategies
with parameter transformations and QoI computation.
"""

import numpy as np
import warnings
from typing import List, Callable, Optional, Dict, Tuple
import matplotlib.pyplot as plt

from .distributions import DistributionFactory
from .doe_strategies import DOEFactory


def _qoi_preview(obj, *, max_chars: int = 500) -> str:
    """Return a short, terminal-friendly preview of a QoI value."""

    def _arr_desc(a: np.ndarray) -> str:
        a = np.asarray(a)
        # Keep output stable and short: shape/dtype + tiny preview
        with np.printoptions(edgeitems=2, threshold=8, linewidth=120, suppress=True):
            preview = np.array2string(a)
        return f"array(shape={a.shape}, dtype={a.dtype}, preview={preview})"

    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            try:
                av = np.asarray(v)
                if av.ndim == 0:
                    desc = f"scalar(dtype={av.dtype}, value={av.item()})"
                else:
                    desc = f"array(shape={av.shape}, dtype={av.dtype})"
            except Exception:
                desc = f"{type(v).__name__}"
            parts.append(f"{k!r}: {desc}")
        s = "{" + ", ".join(parts) + "}"
    else:
        try:
            a = np.asarray(obj)
            if a.ndim == 0:
                s = f"scalar(dtype={a.dtype}, value={a.item()})"
            else:
                s = _arr_desc(a)
        except Exception:
            s = repr(obj)

    if len(s) <= max_chars:
        return s
    return s[: max_chars - 40] + f" ... (truncated, {len(s)} chars)"


_DEFAULT_QOI_FLAT_KEY = "K"


class sampler_legacy_cls:
    """

    Attributes:
        distributions: List of distribution types for each parameter
        bounds: List of distribution parameters for each parameter
        active_keys: Optional list of parameter names
            QoIs: Callable that computes QoI from parameters
        plot_solution: Optional callable for visualization
        n_out: Number of QoI outputs (auto-detected if None)
    """

    def __init__(
        self,
        distributions: List[str],
        bounds: List[list],
        active_keys: Optional[List[str]] = None,
        qoi_fn: Optional[Callable] = None,
        compute_QoIs: Optional[Callable] = None,
        plot_solution: Optional[Callable] = None,
        n_out: Optional[int] = None,
        seed: Optional[int] = None,
        doe_type: str = 'LHS',
        DOE_type: Optional[str] = None,
    ):
        """
        Initialize sampler with distribution specifications.

        Args:
            distributions: List of distribution types ('uniform', 'log_uniform', 'normal', 'lognormal')
            bounds: List of [param1, param2] for each dimension
            active_keys: Optional parameter names
                qoi_fn: Function(params_dict) -> QoI array
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
        # Backward-compatible alias: prefer `qoi_fn`, fall back to legacy `compute_QoIs`
        self.qoi_fn = qoi_fn if qoi_fn is not None else compute_QoIs
        self.compute_QoIs = self.qoi_fn  # expose legacy name
        self.seed = int(seed) if seed is not None else None
        self.plot_solution = plot_solution
        if n_out is not None:
            warnings.warn(
                "`n_out` is deprecated and ignored; QoI output size is auto-detected.",
                DeprecationWarning,
            )
        self.n_out = None
        # backward compatibility: accept DOE_type as legacy keyword
        self.doe_type = DOE_type or doe_type
        self.DOE_type = self.doe_type  # backward-compatible attribute name

        # Create distribution strategy instances
        self.distribution_strategies = [DistributionFactory.create(t) for t in distributions]

        # Print initialization info
        print("Building your FEA sampler...")
        for i, (dist_type, bound) in enumerate(zip(self.distributions, self.bounds)):
            if self.active_keys is not None:
                print(f"parameter dimension {i}: {self.active_keys[i]}")
            print(f"distribution type for dimension {i}: {dist_type}")
            print(f"bounds of distribution for dimension {i}: {bound}")

        # Always auto-detect n_out when a QoI function is available
        if self.qoi_fn is not None:
            self._detect_n_out()
            print(f"n_out: {self.n_out}")

        # Storage for samples (X is computed on-demand from X_reference)
        self.X_reference: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.sampler_doe: Optional[object] = None

        print("... done building")

    def _is_sparse_grid(self) -> bool:
        """Check if current DOE type is sparse grid (SG)."""
        return (self.doe_type is not None and str(self.doe_type).upper() == 'SG') or (
            getattr(self, 'DOE_type', None) is not None and str(self.DOE_type).upper() == 'SG'
        )

    def _resolve_n_samples(self, n_samples: Optional[int], N: Optional[int], **kwargs) -> int:
        """Resolve n_samples from multiple parameter sources."""
        if n_samples is None:
            if N is None:
                n_samples = kwargs.get("n_samples", None) or kwargs.get("N", None)
                if n_samples is None:
                    raise ValueError("n_samples (or legacy N) must be provided")
            else:
                n_samples = N
        return n_samples

    def _resolve_level_for_sg(self, level: Optional[int], n_samples: Optional[int], N: Optional[int]) -> int:
        """Resolve refinement level for sparse grid DOE."""
        if level is None:
            if n_samples is not None:
                warnings.warn(
                    "Passing `n_samples` as sparse-grid refinement level is deprecated; use `level=` when doe_type='SG'.",
                    DeprecationWarning,
                )
                level = int(n_samples)
            elif N is not None:
                warnings.warn(
                    "Passing `N` as sparse-grid refinement level is deprecated; use `level=` when doe_type='SG'.",
                    DeprecationWarning,
                )
                level = int(N)
            else:
                raise ValueError("doe_type='SG' requires `level` to be specified")
        return level

    @property
    def X(self) -> Optional[np.ndarray]:
        """Denormalized physical-space samples (computed from X_reference on demand)."""
        if self.X_reference is None:
            return None
        return self._reference_to_physical_samples(self.X_reference)

    @property
    def X_normalised(self) -> Optional[np.ndarray]:
        """Normalized/unit-space samples in [-1, 1]^d (legacy spelling)."""
        return self.X_reference

    @property
    def X_normalized(self) -> Optional[np.ndarray]:
        """Normalized/unit-space samples in [-1, 1]^d (US spelling)."""
        return self.X_reference

    # New preferred lowercase aliases with backward-compat read-only mapping
    @property
    def x(self) -> Optional[np.ndarray]:
        return self.X

    @property
    def x_normalised(self) -> Optional[np.ndarray]:
        return self.X_reference

    @property
    def x_normalized(self) -> Optional[np.ndarray]:
        return self.X_reference

    # Reference/unit-space accessors for legacy sampler
    @property
    def x_reference(self) -> Optional[np.ndarray]:
        return self.X_reference

    @property
    def x_unit(self) -> Optional[np.ndarray]:
        return self.X_reference

    @property
    def X_unit(self) -> Optional[np.ndarray]:
        return self.X_reference

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs by calling `qoi_fn` at center point."""
        print("n_out not provided -> calling FEA solver for automatic determination")

        X = self._get_center_point()

        if self.active_keys is not None:
            params_dict = {key: X[0, i] for i, key in enumerate(self.active_keys)}
            qval = self.qoi_fn(params_dict)
        else:
            qval = self.qoi_fn(X)

        print(f"QoIs at test point: {_qoi_preview(qval)}")
        arr = np.asarray(qval)
        # Scalar QoI -> one output
        if arr.ndim == 0:
            self.n_out = 1
        else:
            self.n_out = int(np.max(arr.shape))

    def _get_center_point(self) -> np.ndarray:
        """Get center point in physical space based on distribution types."""
        X = np.zeros((1, len(self.bounds)))

        for i, (dist_type, param) in enumerate(zip(self.distributions, self.bounds)):

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
        n_samples: Optional[int] = None,
        N: Optional[int] = None,
        level: Optional[int] = None,
        as_additional_samples: bool = True,
        as_additional_points: Optional[bool] = None,
        plot: bool = False,
        sample_in_batch: bool = False,
        **kwargs,
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
        # Handle sparse grid DOE first - it may use level instead of n_samples
        is_sg = self._is_sparse_grid()
        
        if is_sg and level is not None:
            # For SG with explicit level, we don't need n_samples
            n_samples = None
        else:
            # Resolve n_samples parameter for non-SG or SG without explicit level
            n_samples = self._resolve_n_samples(n_samples, N, **kwargs)

        if as_additional_points is not None:
            as_additional_samples = as_additional_points
        # accept legacy name passed in kwargs
        if "as_additional_points" in kwargs and not as_additional_samples:
            as_additional_samples = kwargs.get("as_additional_points")

        # Resolve level for sparse grid
        if is_sg:
            level = self._resolve_level_for_sg(level, n_samples, N)
            n_samples = None  # Use level instead for SG

        if self.sampler_doe is None or not as_additional_samples:
            if self.sampler_doe is not None and not as_additional_samples:
                print(
                    "Warning: Sampling is being reinitialized. Set as_additional_samples=True to continue sampling."
                )
            try:
                self.sampler_doe = DOEFactory.create(self.doe_type, len(self.bounds), seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            # For SG, pass `level` through as the first arg (legacy API expects n_samples==level)
            if is_sg:
                X_reference = self.sampler_doe.sample(level, as_additional_samples=False)
            else:
                X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=False)
        else:
            # Reuse existing DOE sampler and append points
            if is_sg:
                X_reference = self.sampler_doe.sample(level, as_additional_samples=True)
            else:
                X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=True)

        # Denormalize samples
        X = self._reference_to_physical_samples(X_reference)

        # Compute QoIs
        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        # Build informative sampling message
        n_before = len(self.X_reference) if self.X_reference is not None else 0
        n_new = len(X_reference)
        n_after = n_before + n_new
        
        if self.sampler_doe is not None and not as_additional_samples and n_before > 0:
            # Reinitialization case
            print(
                f"Start computing {n_new} samples in parametric dimension {len(self.bounds)} using {self.doe_type} "
                f"(reinitializing, previous {n_before} samples will be replaced)"
            )
        elif n_before == 0:
            # First call case
            print(
                f"Start computing {n_new} samples in parametric dimension {len(self.bounds)} using {self.doe_type}"
            )
        else:
            # Additional samples case
            print(
                f"Start computing {n_new} additional samples in parametric dimension {len(self.bounds)} using {self.doe_type} "
                f"({n_before} -> {n_after} total samples)"
            )

        # Evaluate samples
        if sample_in_batch and self.qoi_fn is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X, plot)

        print("... done sampling")

        # Store results - use the resolved as_additional_samples value
        self._store_results(X_reference, X, Y, as_additional_samples)

    def _reference_to_physical_samples(self, X_reference: np.ndarray) -> np.ndarray:
        """Transform samples from reference space [-1,1]^n to physical space."""
        return self._transform_samples(X_reference, lambda j, vals: self.distribution_strategies[j].denormalise(vals, self.bounds[j]))

    def _physical_to_reference_samples(self, X: np.ndarray) -> np.ndarray:
        """Transform samples from physical space to reference [-1,1]^n."""
        return self._transform_samples(X, lambda j, vals: self.distribution_strategies[j].normalise(vals, self.bounds[j]))

    def _transform_samples(self, X: np.ndarray, transform_fn: Callable) -> np.ndarray:
        """Apply transformation function to each dimension of samples."""
        result = np.zeros_like(X)
        for j in range(len(self.bounds)):
            result[:, j] = transform_fn(j, X[:, j])
        return result

    # ========================================================================
    # CANONICAL COORDINATE TRANSFORMATION METHODS
    # Convention: phys_to_unit, unit_to_phys (reference/unit are equivalent)
    # All other names are aliases for backward compatibility
    # ========================================================================

    def phys_to_unit(self, x_or_X, *, clip=False):
        """Convert from physical space to unit/reference space [-1, 1].
        
        This is the canonical method. Aliases: physical_to_reference, phys_to_reference.
        """
        x = np.atleast_1d(x_or_X)
        is_1d = (x_or_X.ndim == 0 or (isinstance(x_or_X, np.ndarray) and x_or_X.ndim == 1))
        
        if is_1d:
            x = x.reshape(1, -1)
        
        result = self._transform_samples(x, lambda j, vals: self.distribution_strategies[j].normalise(vals, self.bounds[j]))
        
        if is_1d:
            return result[0]
        return result

    def unit_to_phys(self, z_or_Z, *, clip=False):
        """Convert from unit/reference space [-1, 1] to physical space.
        
        This is the canonical method. Aliases: reference_to_physical, unit_to_physical.
        """
        z = np.atleast_1d(z_or_Z)
        is_1d = (z_or_Z.ndim == 0 or (isinstance(z_or_Z, np.ndarray) and z_or_Z.ndim == 1))
        
        if is_1d:
            z = z.reshape(1, -1)
        
        result = self._transform_samples(z, lambda j, vals: self.distribution_strategies[j].denormalise(vals, self.bounds[j]))
        
        if is_1d:
            return result[0]
        return result

    # ========================================================================
    # BACKWARD-COMPATIBLE ALIASES
    # These all delegate to the canonical methods above
    # ========================================================================

    def physical_to_reference(self, X: np.ndarray) -> np.ndarray:
        """Alias for phys_to_unit."""
        return self.phys_to_unit(X)

    def reference_to_physical(self, X_reference: np.ndarray) -> np.ndarray:
        """Alias for unit_to_phys."""
        return self.unit_to_phys(X_reference)

    def phys_to_reference(self, x_or_X, *, clip=False):
        """Alias for phys_to_unit."""
        return self.phys_to_unit(x_or_X, clip=clip)

    def reference_to_phys(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def unit_to_physical(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def unit_to_phy(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys (short name)."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def phys_to_gauss(self, x_or_X, *, clip=False, eps=None):
        """Legacy sampler has no gaussian-space transform support."""
        raise NotImplementedError(
            "Gaussian-space transforms are not available in sampler_legacy_cls; "
            "use the params-based sampler_cls with a params processor that implements gaussian transforms."
        )

    def gauss_to_phys(self, g_or_G, *, clip=False):
        """Legacy sampler has no gaussian-space transform support."""
        raise NotImplementedError(
            "Gaussian-space transforms are not available in sampler_legacy_cls; "
            "use the params-based sampler_cls with a params processor that implements gaussian transforms."
        )

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate all samples at once."""
        if self.qoi_fn is None:
            return None

        if self.active_keys is not None:
            raise NotImplementedError("Batch evaluation not yet implemented for dimension mapping")

        try:
            return self.qoi_fn(X)
        except TypeError as e:
            raise TypeError(
                f"QoI evaluation failed: {e}. "
                f"Expected array input, but the QoI might be expecting a dict (active_keys={self.active_keys}). "
                f"Ensure QoI signature matches the input type."
            ) from e

    def _evaluate_sequential(self, X: np.ndarray, plot: bool = False) -> Optional[np.ndarray]:
        """Evaluate samples sequentially with optional visualization."""
        if self.qoi_fn is None:
            return None

        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None

        for i in range(len(X)):
            try:
                if self.active_keys is not None:
                    params_dict = {key: X[i, j] for j, key in enumerate(self.active_keys)}
                    qval = self.qoi_fn(params_dict)
                else:
                    qval = self.qoi_fn(X[i, :].reshape(1, -1))
                
                # Auto-detect n_out on first sample if not set
                if Y is None:
                    self.n_out = np.max(np.asarray(qval).shape)
                    Y = np.zeros((len(X), self.n_out))
                
                Y[i, :] = qval
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
        if self.X_reference is None or not as_additional_points:
            self.X_reference = X_reference
            self.Y = Y
        else:
            self.X_reference = np.concatenate((self.X_reference, X_reference), axis=0)
            if Y is not None:
                if self.Y is not None:
                    self.Y = np.concatenate((self.Y, Y), axis=0)
                else:
                    # First time computing Y in additional samples mode
                    self.Y = Y

    @property
    def y(self) -> Optional[np.ndarray]:
        return self.Y
    

# Backwards-compatible aliases will be finalized at EOF

# Backwards-compatible aliases for historical class names
# Older tests and code may import `sampler_new_cls` / `sampler_old_cls`.
# These are set to the concrete classes at the end of the module
# after both `sampler_legacy_cls` and `sampler_cls` are defined.






class sampler_cls:
    """Sampler that uses an external `params` processor for unit<->physical transforms.

    The `params` argument must implement the same API as `params_cls` (e.g. `reference_to_physical`,
    `physical_to_reference`, `pack`, `unpack`, and `dim`). This sampler delegates normalization
    logic to the provided `params` instance instead of keeping its own distribution strategies.
    """

    def __init__(
        self,
        params,
        qoi_fn: Optional[Callable] = None,
        compute_QoIs: Optional[Callable] = None,
        active_keys: Optional[List[str]] = None,
        seed: Optional[int] = None,
        doe_type: str = 'LHS',
        DOE_type: Optional[str] = None,
        qoi_receive_packed: bool = False,
        qoi_force_2d: bool = False,
        qoi_flat_key: Optional[str] = None,
    ):
        # params is expected to encapsulate parameter layout and optionally
        # metadata like n_out. Accept qoi_fn and active_keys as explicit
        # constructor args that override any attributes on the params object.
        self.params = params
        # Backward-compatible alias: prefer `qoi_fn`, fall back to legacy `compute_QoIs`
        self.qoi_fn = qoi_fn if qoi_fn is not None else compute_QoIs
        self.compute_QoIs = self.qoi_fn  # expose legacy name
        self.active_keys = active_keys if active_keys is not None else getattr(params, "active_keys", None)
        # If True, ensure all array inputs passed to QoI are 2D (M, d), even for single samples.
        # This helps with ML models (sklearn/torch) that expect 2D inputs.
        self.qoi_force_2d = bool(qoi_force_2d)
        # If set, non-dict QoI outputs are treated as if they were returned
        # as a single entry in a dict under this key (enables qoi_slices/qoi_indices).
        self.qoi_flat_key = qoi_flat_key
        self.seed = int(seed) if seed is not None else None
        # If False (default), expand packed representations to the full base
        # parameter array before calling `qoi_fn` so QoIs see the same full
        # parameter shape as dict-mode processors. Set True to receive packed.
        self.qoi_receive_packed = bool(qoi_receive_packed)
        # plotting is not supported by sampler_new_cls; omit plot_solution
        # n_out is not stored on params; detect from the provided `qoi_fn` when needed
        self.n_out = None
        # backward compatibility for constructor kwarg
        self.doe_type = DOE_type or doe_type
        self.DOE_type = self.doe_type  # backward-compatible attribute name

        # Use the parameter processor's dimension
        self.dim = int(self.params.dim)

        # DOE sampler placeholder
        self.sampler_doe: Optional[object] = None

        print("Building sampler_new_cls using external params processor...")
        print(f"param processor dim: {self.dim}")
        if self.active_keys is not None:
            for i, k in enumerate(self.active_keys):
                print(f"parameter dimension {i}: {k}")

        if self.qoi_fn is not None:
            self._detect_n_out()
            print(f"n_out: {self.n_out}")

        # Storage for samples
        self.X_reference: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None

        print("... done building sampler_new_cls")

    def _effective_qoi_flat_key(self) -> str:
        """Return the key used for single-output (non-dict) QoIs.

        Always returns a key so that a QoI layout can be built and unflattening
        is available without extra user configuration.
        """
        return self.qoi_flat_key if self.qoi_flat_key is not None else _DEFAULT_QOI_FLAT_KEY

    def _is_sparse_grid(self) -> bool:
        """Check if current DOE type is sparse grid (SG)."""
        return (self.doe_type is not None and str(self.doe_type).upper() == 'SG') or (
            getattr(self, 'DOE_type', None) is not None and str(self.DOE_type).upper() == 'SG'
        )

    def _resolve_n_samples(self, n_samples: Optional[int], N: Optional[int], **kwargs) -> int:
        """Resolve n_samples from multiple parameter sources."""
        if n_samples is None:
            if N is None:
                n_samples = kwargs.get('n_samples') or kwargs.get('N')
                if n_samples is None:
                    raise ValueError("n_samples (or legacy N) must be provided")
            else:
                n_samples = N
        return n_samples

    def _resolve_level_for_sg(self, level: Optional[int], n_samples: Optional[int], N: Optional[int]) -> int:
        """Resolve refinement level for sparse grid DOE."""
        if level is None:
            if n_samples is not None:
                warnings.warn(
                    "Passing `n_samples` as sparse-grid refinement level is deprecated; use `level=` when doe_type='SG'.",
                    DeprecationWarning,
                )
                level = int(n_samples)
            elif N is not None:
                warnings.warn(
                    "Passing `N` as sparse-grid refinement level is deprecated; use `level=` when doe_type='SG'.",
                    DeprecationWarning,
                )
                level = int(N)
            else:
                raise ValueError("doe_type='SG' requires `level` to be specified")
        return level

    @property
    def X(self) -> Optional[np.ndarray]:
        if self.X_reference is None:
            return None
        return self.unit_to_phys(self.X_reference)

    @property
    def x(self) -> Optional[np.ndarray]:
        return self.X

    # Reference/unit-space sample accessors (canonical names)
    @property
    def x_unit(self) -> Optional[np.ndarray]:
        return self.X_reference

    @property
    def x_reference(self) -> Optional[np.ndarray]:
        return self.x_unit

    @property
    def X_unit(self) -> Optional[np.ndarray]:
        return self.x_unit

    @property
    def X_gaussian(self) -> Optional[np.ndarray]:
        """Return samples in gaussian space corresponding to `X_reference`.

        Delegates conversion to the injected `params` processor (expects
        a `unit_to_gauss` or `reference_to_gauss`-style method). Returns
        None if no reference samples are stored.
        """
        if self.X_reference is None:
            return None
        # prefer unit_to_gauss (compatibility name) on the params processor
        if hasattr(self.params, "unit_to_gauss"):
            return self.params.unit_to_gauss(self.X_reference)
        # fallback if someone provided a reference->gauss named differently
        if hasattr(self.params, "reference_to_gauss"):
            return self.params.reference_to_gauss(self.X_reference)
        raise AttributeError("params processor has no method to convert reference -> gaussian (expected 'unit_to_gauss' or 'reference_to_gauss')")

    # Canonical lowercase alias for gaussian samples
    @property
    def x_gauss(self) -> Optional[np.ndarray]:
        return self.X_gaussian

    # Backwards-compatible aliases
    @property
    def x_gaussian(self) -> Optional[np.ndarray]:
        return self.X_gaussian

    @property
    def X_gauss(self) -> Optional[np.ndarray]:
        return self.X_gaussian

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs using the params base point."""
        print("n_out not provided -> calling QoI for automatic determination")
        x_center = self.params.pack(self.params.base)
        # Prefer named dict input when the params processor provides an unpack method
        if self.active_keys is not None or hasattr(self.params, "unpack"):
            unpacked = self.params.unpack(x_center)
            # If params.processor is array-mode and we prefer full arrays,
            # expand the packed representation to the full base array.
            if isinstance(unpacked, np.ndarray) and not self.qoi_receive_packed:
                # build full array
                base_arr = np.asarray(self.params.base.get("__arr", [])).copy()
                if base_arr.size:
                    full = base_arr.copy()
                    for it in getattr(self.params, "_layout", []):
                        if it.get("param") == "__arr":
                            sl = it["sl"]
                            mask = it["mask"]
                            full[mask] = unpacked[sl]
                    if self.qoi_force_2d and np.asarray(full).ndim == 1:
                        qval = self.qoi_fn(np.asarray(full).reshape(1, -1))
                    else:
                        qval = self.qoi_fn(full)
                else:
                    if self.qoi_force_2d and np.asarray(unpacked).ndim == 1:
                        qval = self.qoi_fn(np.asarray(unpacked).reshape(1, -1))
                    else:
                        qval = self.qoi_fn(unpacked)
            else:
                if isinstance(unpacked, np.ndarray) and self.qoi_force_2d and np.asarray(unpacked).ndim == 1:
                    qval = self.qoi_fn(np.asarray(unpacked).reshape(1, -1))
                else:
                    qval = self.qoi_fn(unpacked)
        else:
            if self.qoi_force_2d and np.asarray(x_center).ndim == 1:
                qval = self.qoi_fn(np.asarray(x_center).reshape(1, -1))
            else:
                qval = self.qoi_fn(x_center)
        print(f"QoIs at test point: {_qoi_preview(qval)}")
        # If QoIs returns a dict, build layout to allow slicing by key
        if isinstance(qval, dict):
            self._build_qoi_layout(qval)
            self.n_out = self.qoi_dim
        else:
            arr = np.asarray(qval)
            self.qoi_dim = int(arr.size) if arr.size is not None else int(np.max(arr.shape))
            key = self._effective_qoi_flat_key()
            self._qoi_layout = [
                {
                    "key": key,
                    "shape": arr.shape,
                    "size": int(self.qoi_dim),
                    "sl": slice(0, int(self.qoi_dim)),
                }
            ]
            self.n_out = self.qoi_dim

    def _build_qoi_layout(self, qoi_dict: dict) -> None:
        """Build layout mapping QoI dict keys to flattened index ranges."""
        layout = []
        cursor = 0
        for key, val in qoi_dict.items():
            arr = np.asarray(val)
            size = int(arr.size)
            sl = slice(cursor, cursor + size)
            layout.append({"key": key, "shape": arr.shape, "size": size, "sl": sl})
            cursor += size
        self._qoi_layout = layout
        self.qoi_dim = cursor

    def ravel_qoi_dict(self, qoi_dict: dict, keys=None) -> np.ndarray:
        """Flatten a QoI output dict into a 1D vector.

        This mirrors the internal dict-flattening used during sequential/batch
        QoI evaluation.

        Ordering:
        - If a QoI layout exists (built during `_detect_n_out()` or from prior
          samples), that key order is used.
        - Otherwise, insertion order of `qoi_dict` is used and a layout is built
          from this dict.

        Args:
            qoi_dict: Mapping key -> scalar/array-like QoI value for a *single* sample.
            keys: Optional explicit key order (string or sequence). If provided,
                overrides stored layout ordering.

        Returns:
            1D numpy array containing all values concatenated.
        """
        if not isinstance(qoi_dict, dict):
            raise TypeError(f"Expected a dict, got {type(qoi_dict).__name__}")

        layout = getattr(self, "_qoi_layout", None)

        if keys is None:
            if layout is not None:
                key_order = [it["key"] for it in layout]
            else:
                # Fall back to insertion order and build a layout for consistency.
                key_order = list(qoi_dict.keys())
                self._build_qoi_layout(qoi_dict)
        else:
            if isinstance(keys, str):
                key_order = [keys]
            else:
                key_order = list(keys)

        parts = [np.asarray(qoi_dict[k]).ravel() for k in key_order]
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    def unravel_qoi(self, y_flat, keys=None, *, as_dict: Optional[bool] = None, squeeze: bool = True):
        """Unflatten a QoI vector back to its original tensor/dict structure.

        This is the inverse of the internal QoI flattening used by the sampler.
        It relies on the stored QoI layout (`self._qoi_layout`), which is built
        when the QoI returns a dict (or when `qoi_flat_key` is used for non-dict
        outputs).

        Supports:
        - Single sample: `y_flat` shape `(qoi_dim,)`
        - Batched: `y_flat` shape `(M, qoi_dim)`

        Args:
            y_flat: Flattened QoI array.
            keys: Optional key or sequence of keys to extract.
            as_dict: If True, always returns a dict. If False, and the layout has
                a single key (typically `qoi_flat_key`), returns the unflattened
                array for that key. If None (default), returns an array only for
                the single-key (`qoi_flat_key`) case, otherwise returns a dict.
            squeeze: If True, scalar per-sample outputs are squeezed to `(M,)` in
                batch mode and to Python scalars in single-sample mode.

        Returns:
            Dict of key -> unflattened array/tensor, or a single array/tensor for
            the one-key layout.
        """
        layout = getattr(self, "_qoi_layout", None)
        if layout is None:
            raise KeyError(
                "QoI layout not available; cannot unflatten. Ensure the QoI returned a dict at least once, "
                "or set qoi_flat_key for non-dict QoI outputs."
            )

        a = np.asarray(y_flat)
        if a.ndim not in (1, 2):
            raise ValueError(f"Expected y_flat with ndim 1 or 2, got shape {a.shape}")

        qoi_dim = int(getattr(self, "qoi_dim", 0) or (layout[-1]["sl"].stop if layout else 0))
        if a.ndim == 1:
            if a.size != qoi_dim:
                raise ValueError(f"Expected y_flat of length {qoi_dim}, got {a.size}")
        else:
            if a.shape[1] != qoi_dim:
                raise ValueError(f"Expected y_flat with shape (M, {qoi_dim}), got {a.shape}")

        if keys is None:
            key_order = [it["key"] for it in layout]
        else:
            if isinstance(keys, str):
                key_order = [keys]
            else:
                key_order = list(keys)

        def _find_item(k: str) -> dict:
            for it in layout:
                if it["key"] == k:
                    return it
            raise KeyError(f"QoI key '{k}' not found")

        one_key_layout = (len(key_order) == 1)
        if as_dict is None:
            # Default: if caller asked for a single key and that key matches the
            # flat-key layout, return the array directly.
            effective_key = self._effective_qoi_flat_key()
            as_dict_resolved = not (
                one_key_layout
                and len(layout) == 1
                and layout[0]["key"] == effective_key
            )
        else:
            as_dict_resolved = bool(as_dict)

        out_dict = {}
        if a.ndim == 1:
            for k in key_order:
                it = _find_item(k)
                sl = it["sl"]
                shape = tuple(np.atleast_1d(it.get("shape", ())).tolist()) if it.get("shape", ()) != () else ()
                part = np.asarray(a[sl])
                if squeeze and part.size == 1:
                    # Treat 1-element values as scalars (even if recorded shape was (1,))
                    out_dict[k] = part.reshape(()).item()
                else:
                    out_dict[k] = part.reshape(shape)
        else:
            m = int(a.shape[0])
            for k in key_order:
                it = _find_item(k)
                sl = it["sl"]
                shape = tuple(np.atleast_1d(it.get("shape", ())).tolist()) if it.get("shape", ()) != () else ()
                part = np.asarray(a[:, sl])
                if squeeze and part.shape[1] == 1:
                    # Treat 1-element outputs as scalars per sample
                    out_dict[k] = part.reshape(m)
                else:
                    out_dict[k] = part.reshape((m,) + shape)

        if as_dict_resolved:
            return out_dict
        # Return single array/tensor for one-key selection
        return out_dict[key_order[0]]

    def qoi_slices(self, keys):
        """Return slice(s) into the flattened QoI vector for key(s).

        If `keys` is a string, returns a single `slice` for that key. If
        `keys` is a sequence, returns a dict mapping each key to its slice.
        Raises KeyError if the layout is unavailable or a key is missing.
        """
        if self._qoi_layout is None:
            raise KeyError("QoI layout not available (QoIs did not return a dict during detection)")

        def _single(k: str) -> slice:
            for item in self._qoi_layout:
                if item["key"] == k:
                    return item["sl"]
            raise KeyError(f"QoI key '{k}' not found")

        if isinstance(keys, str):
            return _single(keys)
        return {k: _single(k) for k in keys}

    def qoi_indices(self, keys):
        """Return integer index array(s) for QoI key(s).

        If `keys` is a string, returns a 1D `np.ndarray` of indices for that
        key. If `keys` is a sequence, returns a concatenated 1D `np.ndarray`
        of indices in the order of `keys` provided.
        """
        if isinstance(keys, str):
            sl = self.qoi_slices(keys)
            return np.arange(sl.start, sl.stop)

        mapping = self.qoi_slices(keys)
        parts = [np.arange(mapping[k].start, mapping[k].stop) for k in keys]
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)


    def _expand_packed_to_full(self, packed):
        """Expand packed representation(s) returned by params.unpack into the
        full base parameter array(s). Accepts 1D (single) or 2D (batch) packed
        arrays and returns the expanded full array(s).
        """
        base_arr = np.asarray(self.params.base.get("__arr", [])).copy()
        if base_arr.size == 0:
            # Nothing to expand to; return input as-is
            return packed

        a = np.asarray(packed)
        is_1d = (a.ndim == 1)
        
        # Normalize to 2D for uniform processing
        if is_1d:
            a = a.reshape(1, -1)
        
        # Create output array: replicate base array for each sample
        full = np.tile(base_arr[None, :], (a.shape[0], 1))
        
        # Apply transformations from layout
        for it in getattr(self.params, "_layout", []):
            if it.get("param") == "__arr":
                sl = it["sl"]
                mask = it["mask"]
                full[:, mask] = a[:, sl]
        
        # Return in original dimensionality
        return full[0] if is_1d else full

    

    def sample(
        self,
        n_samples: Optional[int] = None,
        N: Optional[int] = None,
        level: Optional[int] = None,
        add_to_dataset: bool = True,
        as_additional_samples: Optional[bool] = None,
        as_additional_points: Optional[bool] = None,
        plot: bool = False,
        batch_computation: bool = False,
        **kwargs,
    ) -> None:
        # Backward-compatible alias: some older callsites used `sample_in_batch`
        # to request batched QoI evaluation.
        if "sample_in_batch" in kwargs:
            batch_computation = bool(kwargs.get("sample_in_batch"))

        # Handle sparse grid DOE first - it may use level instead of n_samples
        is_sg = self._is_sparse_grid()
        
        if is_sg and level is not None:
            # For SG with explicit level, we don't need n_samples
            n_samples = None
        else:
            # Resolve n_samples parameter for non-SG or SG without explicit level
            n_samples = self._resolve_n_samples(n_samples, N, **kwargs)

        # Resolve add-to-dataset flag with backward-compatible aliases.
        # Precedence: explicit `as_additional_samples` if provided -> `add_to_dataset` -> `as_additional_points` kw
        if as_additional_samples is not None:
            as_additional = bool(as_additional_samples)
        else:
            # prefer new name unless legacy as_additional_points provided
            if as_additional_points is not None:
                as_additional = bool(as_additional_points)
            else:
                # allow legacy kw in kwargs to override
                if 'as_additional_points' in kwargs:
                    as_additional = bool(kwargs.get('as_additional_points'))
                else:
                    as_additional = bool(add_to_dataset)

        # Resolve level for sparse grid
        if is_sg:
            level = self._resolve_level_for_sg(level, n_samples, N)
            n_samples = None  # Use level instead for SG

        if self.sampler_doe is None or not as_additional:
            if self.sampler_doe is not None and not as_additional:
                print("Warning: Sampling is being reinitialized. Set as_additional_samples=True to continue sampling.")
            try:
                self.sampler_doe = DOEFactory.create(self.doe_type, self.dim, seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            if is_sg:
                X_reference = self.sampler_doe.sample(level, as_additional_samples=False)
            else:
                X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=False)
        else:
            if is_sg:
                X_reference = self.sampler_doe.sample(level, as_additional_samples=True)
            else:
                X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=True)

        # Convert reference -> physical using params processor
        X = self.unit_to_phys(X_reference)

        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        # Build informative sampling message
        n_before = len(self.X_reference) if self.X_reference is not None else 0
        n_new = len(X_reference)
        n_after = n_before + n_new
        
        if self.sampler_doe is not None and not as_additional and n_before > 0:
            # Reinitialization case
            print(
                f"Start computing {n_new} samples in parametric dimension {self.dim} using {self.doe_type} "
                f"(reinitializing, previous {n_before} samples will be replaced)"
            )
        elif n_before == 0:
            # First call case
            print(
                f"Start computing {n_new} samples in parametric dimension {self.dim} using {self.doe_type}"
            )
        else:
            # Additional samples case
            print(
                f"Start computing {n_new} additional samples in parametric dimension {self.dim} using {self.doe_type} "
                f"({n_before} -> {n_after} total samples)"
            )

        # Decide batch vs sequential based on params processor mode and caller preference
        can_batch = getattr(self.params, "_mode", None) == "array"
        if batch_computation and can_batch and self.qoi_fn is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X)

        print("... done sampling")

        self._store_results(X_reference, X, Y, as_additional_points if as_additional_points is not None else as_additional)

    # ========================================================================
    # CANONICAL COORDINATE TRANSFORMATION METHODS
    # Convention: phys_to_unit, unit_to_gauss, gauss_to_unit, unit_to_phys
    # All other names are aliases for backward compatibility
    # ========================================================================

    def phys_to_unit(self, x_or_X, *, clip=False):
        """Convert from physical space to unit/reference space [-1, 1].
        
        This is the canonical method. Aliases: physical_to_reference, phys_to_reference.
        """
        if hasattr(self.params, "physical_to_reference"):
            return self.params.physical_to_reference(x_or_X, clip=clip)
        if hasattr(self.params, "phys_to_unit"):
            return self.params.phys_to_unit(x_or_X, clip=clip)
        raise AttributeError("params processor has no method 'physical_to_reference' or 'phys_to_unit'")

    def unit_to_phys(self, z_or_Z, *, clip=False):
        """Convert from unit/reference space [-1, 1] to physical space.
        
        This is the canonical method. Aliases: reference_to_physical, unit_to_physical.
        """
        if hasattr(self.params, "reference_to_physical"):
            return self.params.reference_to_physical(z_or_Z, clip=clip)
        if hasattr(self.params, "unit_to_physical"):
            return self.params.unit_to_physical(z_or_Z, clip=clip)
        raise AttributeError("params processor has no method 'reference_to_physical' or 'unit_to_physical'")

    def unit_to_phy(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys (short name)."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def unit_to_gauss(self, z_or_Z, *, eps=None):
        """Convert from unit/reference space [-1, 1] to Gaussian space.
        
        This is the canonical method. Alias: reference_to_gaussian.
        """
        if hasattr(self.params, "unit_to_gauss"):
            return self.params.unit_to_gauss(z_or_Z, eps=eps) if eps is not None else self.params.unit_to_gauss(z_or_Z)
        raise AttributeError("params processor has no method 'unit_to_gauss'")

    def gauss_to_unit(self, g_or_G):
        """Convert from Gaussian space to unit/reference space [-1, 1].
        
        This is the canonical method. Alias: gaussian_to_reference.
        """
        if hasattr(self.params, "gauss_to_unit"):
            return self.params.gauss_to_unit(g_or_G)
        raise AttributeError("params processor has no method 'gauss_to_unit'")

    def phys_to_gauss(self, x_or_X, *, clip=False, eps=None):
        """Convert from physical space to Gaussian space.
        
        Composed from phys_to_unit and unit_to_gauss.
        Alias: physical_to_gauss, physical_to_gaussian.
        """
        if hasattr(self.params, "physical_to_gauss"):
            return self.params.physical_to_gauss(x_or_X, clip=clip, eps=eps)
        # Compose: physical -> unit -> gauss
        z = self.phys_to_unit(x_or_X, clip=clip)
        return self.unit_to_gauss(z, eps=eps)

    def gauss_to_phys(self, g_or_G, *, clip=False):
        """Convert from Gaussian space to physical space.
        
        Composed from gauss_to_unit and unit_to_phys.
        Alias: gauss_to_physical, gaussian_to_physical.
        """
        if hasattr(self.params, "gauss_to_physical"):
            return self.params.gauss_to_physical(g_or_G, clip=clip)
        # Compose: gauss -> unit -> physical
        z = self.gauss_to_unit(g_or_G)
        return self.unit_to_phys(z, clip=clip)

    # ========================================================================
    # BACKWARD-COMPATIBLE ALIASES
    # These all delegate to the canonical methods above
    # ========================================================================

    def physical_to_reference(self, X: np.ndarray, *, clip=False) -> np.ndarray:
        """Alias for phys_to_unit."""
        return self.phys_to_unit(X, clip=clip)

    def reference_to_physical(self, X_reference: np.ndarray, *, clip=False) -> np.ndarray:
        """Alias for unit_to_phys."""
        return self.unit_to_phys(X_reference, clip=clip)

    def phys_to_reference(self, x_or_X, *, clip=False):
        """Alias for phys_to_unit."""
        return self.phys_to_unit(x_or_X, clip=clip)

    def reference_to_phys(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def unit_to_physical(self, z_or_Z, *, clip=False):
        """Alias for unit_to_phys."""
        return self.unit_to_phys(z_or_Z, clip=clip)

    def physical_to_gauss(self, x_or_X, *, clip=False, eps=None):
        """Alias for phys_to_gauss."""
        return self.phys_to_gauss(x_or_X, clip=clip, eps=eps)

    def gauss_to_physical(self, g_or_G, *, clip=False):
        """Alias for gauss_to_phys."""
        return self.gauss_to_phys(g_or_G, clip=clip)

    def gaussian_to_physical(self, g_or_G, *, clip=False):
        """Alias for gauss_to_phys."""
        return self.gauss_to_phys(g_or_G, clip=clip)

    def gaussian_to_reference(self, g_or_G):
        """Alias for gauss_to_unit."""
        return self.gauss_to_unit(g_or_G)

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        if self.qoi_fn is None:
            return None

        def _normalize_batched_array(out, m_expected: int) -> np.ndarray:
            """Normalize QoI output for batch evaluation.

            Ensures a 2D array of shape (m_expected, n_out).
            Accepts common QoI return shapes:
            - (m_expected,) -> (m_expected, 1)
            - (m_expected, 1) -> (m_expected, 1)
            - (1, m_expected) with n_out==1 -> (m_expected, 1)
            - (m_expected, k, ...) -> (m_expected, k*...)
            - (k,) when m_expected==1 -> (1, k)
            """
            arr = np.asarray(out)
            if arr.ndim == 0:
                raise ValueError(
                    "Batch QoI returned a scalar; expected one value (or vector) per sample"
                )

            # 1D outputs
            if arr.ndim == 1:
                if arr.shape[0] == m_expected:
                    arr2 = arr.reshape(m_expected, 1)
                elif m_expected == 1:
                    arr2 = arr.reshape(1, -1)
                elif self.n_out is not None and arr.size == m_expected * int(self.n_out):
                    arr2 = arr.reshape(m_expected, int(self.n_out))
                else:
                    raise ValueError(
                        f"Batch QoI returned shape {arr.shape}; expected ({m_expected},) or ({m_expected}, n_out)"
                    )
            else:
                # Special-case common row-vector form for scalar outputs
                if (
                    arr.ndim == 2
                    and arr.shape[0] == 1
                    and arr.shape[1] == m_expected
                    and (self.n_out == 1 or self.n_out is None)
                ):
                    arr2 = arr.T
                elif arr.shape[0] == m_expected:
                    arr2 = arr.reshape(m_expected, -1)
                elif m_expected == 1:
                    arr2 = arr.reshape(1, -1)
                else:
                    raise ValueError(
                        f"Batch QoI returned shape {arr.shape}; expected first dimension to be {m_expected}"
                    )

            # Ensure n_out matches
            if self.n_out is None:
                self.n_out = int(arr2.shape[1])
                self.qoi_dim = int(self.n_out)
            else:
                n_out = int(self.n_out)
                if arr2.shape[1] != n_out:
                    raise ValueError(
                        f"Batch QoI returned {arr2.shape[1]} outputs per sample but expected n_out={n_out}"
                    )
            return arr2

        # If params.unpack exists, ask it to convert the packed X to the
        # form expected by the QoI. For array-style processors this will be
        # a (M,N) ndarray and can be passed directly to the QoI function.
        if hasattr(self.params, "unpack"):
            unpacked = self.params.unpack(X)
            # array-style unpack -> pass batched array to QoI
            if isinstance(unpacked, np.ndarray):
                # Decide whether to pass packed or expanded full arrays
                if getattr(self.params, "_mode", None) == "array" and "__arr" in getattr(self.params, "base", {}):
                    if self.qoi_receive_packed:
                        to_pass = unpacked
                    else:
                        to_pass = self._expand_packed_to_full(unpacked)
                else:
                    to_pass = unpacked

                try:
                    out = self.qoi_fn(to_pass)
                except TypeError:
                    # Fallback: QoI may expect single-sample arrays; warn and evaluate sequentially
                    warnings.warn(
                        "Batch QoI evaluation failed; falling back to sequential evaluation",
                        UserWarning,
                    )
                    outs = []
                    for i in range(unpacked.shape[0]):
                        xi = to_pass[i]
                        if self.qoi_force_2d and np.asarray(xi).ndim == 1:
                            xi = np.asarray(xi).reshape(1, -1)
                        oi = self.qoi_fn(xi)
                        outs.append(np.asarray(oi).ravel())
                    return np.vstack(outs)

                # If QoI returned a dict for batched input, flatten accordingly
                if isinstance(out, dict):
                    keys = list(out.keys())
                    parts = []
                    layout = []
                    cursor = 0
                    for k in keys:
                        v = np.asarray(out[k])
                        if v.ndim == 0:
                            v = v.reshape(1, 1)
                        elif v.ndim == 1:
                            v = v.reshape(-1, 1)
                        m = v.shape[0]
                        per_sample_size = int(np.prod(v.shape[1:]))
                        layout.append({"key": k, "shape": v.shape[1:], "size": per_sample_size, "sl": slice(cursor, cursor + per_sample_size)})
                        cursor += per_sample_size
                        parts.append(v.reshape(m, -1))
                    # save layout and qoi_dim
                    self._qoi_layout = layout
                    self.qoi_dim = cursor
                    self.n_out = cursor
                    return np.concatenate(parts, axis=1)

                arr2 = _normalize_batched_array(out, unpacked.shape[0])
                # Always build a one-key layout for non-dict outputs.
                key = self._effective_qoi_flat_key()
                n_out = int(self.n_out) if self.n_out is not None else int(arr2.shape[1])
                self.qoi_dim = int(n_out)
                out_arr = np.asarray(out)
                if out_arr.ndim >= 2 and out_arr.shape[0] == arr2.shape[0]:
                    per_sample_shape = tuple(out_arr.shape[1:])
                elif out_arr.ndim == 1 and out_arr.shape[0] == arr2.shape[0]:
                    per_sample_shape = ()
                else:
                    per_sample_shape = (n_out,)
                self._qoi_layout = [
                    {"key": key, "shape": per_sample_shape, "size": n_out, "sl": slice(0, n_out)}
                ]
                return arr2

            # dict-style unpack not supported in batch mode
            raise NotImplementedError("Batch evaluation not implemented for dict-style QoIs (use sequential evaluation)")

        # No unpacking required; pass X directly
        try:
            out = self.qoi_fn(X)
            if isinstance(out, dict):
                raise NotImplementedError(
                    "Batch evaluation not implemented for dict-style QoIs without params.unpack (use sequential evaluation)"
                )
            arr2 = _normalize_batched_array(out, X.shape[0])
            # Always build a one-key layout for non-dict outputs.
            key = self._effective_qoi_flat_key()
            n_out = int(self.n_out) if self.n_out is not None else int(arr2.shape[1])
            self.qoi_dim = int(n_out)
            out_arr = np.asarray(out)
            # Best-effort preserve original per-sample tensor shape.
            if out_arr.ndim >= 2 and out_arr.shape[0] == X.shape[0]:
                per_sample_shape = tuple(out_arr.shape[1:])
            elif out_arr.ndim == 1 and out_arr.shape[0] == X.shape[0]:
                per_sample_shape = ()
            else:
                per_sample_shape = (n_out,)
            self._qoi_layout = [
                {
                    "key": key,
                    "shape": per_sample_shape,
                    "size": n_out,
                    "sl": slice(0, n_out),
                }
            ]
            return arr2
        except TypeError as e:
            raise TypeError(f"QoI evaluation failed: {e}") from e

    def _evaluate_sequential(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.qoi_fn is None:
            return None
        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None
        for i in range(len(X)):
            try:
                if self.active_keys is not None:
                    params_dict = {key: X[i, j] for j, key in enumerate(self.active_keys)}
                    out = self.qoi_fn(params_dict)
                elif hasattr(self.params, 'unpack'):
                    unpacked = self.params.unpack(X[i])
                    # If unpack returns an ndarray (array-style processor), pass
                    # the flat array to the QoI; otherwise assume dict and pass it.
                    if isinstance(unpacked, np.ndarray):
                        # For array-mode processors, optionally expand packed vector
                        # to the full base array before calling QoI depending on
                        # the qoi_receive_packed flag.
                        if getattr(self.params, "_mode", None) == "array" and "__arr" in getattr(self.params, "base", {}):
                            if self.qoi_receive_packed:
                                to_call = unpacked if unpacked.ndim == 1 else unpacked.reshape(unpacked.shape)
                            else:
                                to_call = self._expand_packed_to_full(unpacked)
                        else:
                            to_call = unpacked if unpacked.ndim == 1 else unpacked.reshape(unpacked.shape)

                        if self.qoi_force_2d and np.asarray(to_call).ndim == 1:
                            to_call = np.asarray(to_call).reshape(1, -1)
                        out = self.qoi_fn(to_call)
                    else:
                        out = self.qoi_fn(unpacked)
                else:
                    out = self.qoi_fn(X[i, :].reshape(1, -1))

                # normalize output to array or dict
                if isinstance(out, dict):
                    # need qoi layout to flatten dict consistently
                    if self._qoi_layout is None:
                        # build layout from this single sample
                        self._build_qoi_layout(out)
                        self.n_out = self.qoi_dim
                        # resize Y if necessary
                        if Y is None:
                            Y = np.zeros((len(X), self.n_out))
                    parts = []
                    for item in self._qoi_layout:
                        key = item['key']
                        val = np.asarray(out[key]).ravel()
                        parts.append(val)
                    Y[i, :] = np.concatenate(parts)
                else:
                    # Auto-detect n_out on first sample if not set
                    if Y is None:
                        out_arr = np.asarray(out).ravel()
                        self.n_out = len(out_arr)
                        self.qoi_dim = int(self.n_out)
                        if getattr(self, "_qoi_layout", None) is None:
                            n_out = int(self.n_out)
                            key = self._effective_qoi_flat_key()
                            self._qoi_layout = [
                                {
                                    "key": key,
                                    "shape": np.asarray(out).shape,
                                    "size": n_out,
                                    "sl": slice(0, n_out),
                                }
                            ]
                        Y = np.zeros((len(X), self.n_out))
                        Y[i, :] = out_arr
                    else:
                        Y[i, :] = np.asarray(out).ravel()
            except TypeError:
                raise
        return Y

    def _store_results(self, X_reference: np.ndarray, X: np.ndarray, Y: Optional[np.ndarray], as_additional_points: bool) -> None:
        if self.X_reference is None or not as_additional_points:
            self.X_reference = X_reference
            self.Y = Y
        else:
            self.X_reference = np.concatenate((self.X_reference, X_reference), axis=0)
            if Y is not None:
                if self.Y is not None:
                    self.Y = np.concatenate((self.Y, Y), axis=0)
                else:
                    # First time computing Y in additional samples mode
                    self.Y = Y

    @property
    def y(self) -> Optional[np.ndarray]:
        return self.Y


# Backwards-compatible aliases (finalized)
sampler_new_cls = sampler_cls
sampler_old_cls = sampler_legacy_cls

# Store reference to the actual class before it gets shadowed
_sampler_cls_class = sampler_cls
_sampler_legacy_cls_class = sampler_legacy_cls


def sampler_cls(*args, **kwargs):
    """Smart factory for sampler classes.

    Behavior:
    - If caller passes a `params=` keyword (or first positional arg looks
      like a params processor), construct and return the params-based
      `sampler_cls` instance.
    - Otherwise, construct and return the legacy `sampler_legacy_cls`.

    This preserves older callsites that do `sampler_cls(distributions=..., bounds=...)`
    while allowing new code to call `sampler_cls(params=P, ...)`.
    """
    # If explicit 'params' kw is provided, prefer new params-based sampler
    if 'params' in kwargs:
        return _sampler_cls_class(*args, **kwargs)

    # If first positional arg looks like a params processor, prefer new sampler
    if len(args) >= 1:
        first = args[0]
        # Heuristic: params processors expose 'pack' and 'reference_to_physical' or 'dim'
        if hasattr(first, 'pack') or hasattr(first, 'reference_to_physical') or hasattr(first, 'dim'):
            return _sampler_cls_class(*args, **kwargs)

    # Fallback to legacy sampler for distributions/bounds style
    return _sampler_legacy_cls_class(*args, **kwargs)


# Convenience alias - also export sampler as the factory
sampler = sampler_cls
    
