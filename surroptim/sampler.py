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


class sampler_cls:
    """
    Sampler for parametric studies with multiple distribution types and DOE strategies.

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
        self.qoi_fn = qoi_fn
        self.seed = int(seed) if seed is not None else None
        self.plot_solution = plot_solution
        self.n_out = n_out
        # backward compatibility: accept DOE_type as legacy keyword
        self.doe_type = DOE_type if DOE_type is not None else doe_type
        # backward-compatible attribute name
        self.DOE_type = self.doe_type

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
        if n_out is None and qoi_fn is not None:
            self._detect_n_out()

        if qoi_fn is not None:
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

    # New preferred lowercase aliases with backward-compat read-only mapping
    @property
    def x(self) -> Optional[np.ndarray]:
        return self.X

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs by calling `qoi_fn` at center point."""
        print("n_out not provided -> calling FEA solver for automatic determination")

        X = self._get_center_point()

        if self.active_keys is not None:
            params_dict = {self.active_keys[i]: X[0, i] for i in range(len(self.active_keys))}
            qval = self.qoi_fn(params_dict)
        else:
            qval = self.qoi_fn(X)

        print(f"QoIs at test point: {qval}")
        self.n_out = np.max(np.asarray(qval).shape)

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
        n_samples: Optional[int] = None,
        N: Optional[int] = None,
        as_additional_samples: bool = False,
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
        # Setup DOE strategy
        # resolve backward-compatible args and accept legacy kwargs
        if n_samples is None:
            # prefer explicit N if provided
            if N is None:
                # accept legacy kw in **kwargs
                n_samples = kwargs.get("n_samples", None) or kwargs.get("N", None)
                if n_samples is None:
                    raise ValueError("n_samples (or legacy N) must be provided")
            else:
                n_samples = N

        if as_additional_points is not None:
            as_additional_samples = as_additional_points
        # accept legacy name passed in kwargs
        if "as_additional_points" in kwargs and not as_additional_samples:
            as_additional_samples = kwargs.get("as_additional_points")

        if self.sampler_doe is None or not as_additional_samples:
            if self.sampler_doe is not None and not as_additional_samples:
                print(
                    "Warning: Sampling is being reinitialized. Set as_additional_samples=True to continue sampling."
                )
            try:
                self.sampler_doe = DOEFactory.create(self.doe_type, len(self.bounds), seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=False)
        else:
            # Reuse existing DOE sampler and append points
            X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=True)

        # Denormalize samples
        X = self._reference_to_physical_samples(X_reference)

        # Compute QoIs
        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        sample_type = "additional samples" if as_additional_samples and self.X_reference is not None else "samples"
        print(
            f"Start computing {len(X_reference)} {sample_type} in parametric dimension {len(self.bounds)} using {self.DOE_type}"
        )

        # Evaluate samples
        if sample_in_batch and self.qoi_fn is not None:
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
        """Transform samples from physical space to reference [-1,1]^n."""
        Z = np.zeros_like(X)
        for j in range(len(self.bounds)):
            Z[:, j] = self.distribution_strategies[j].normalise(X[:, j], self.bounds[j])
        return Z

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
                    params_dict = {self.active_keys[j]: X[i, j] for j in range(len(self.active_keys))}
                    Y[i, :] = self.qoi_fn(params_dict)
                else:
                    Y[i, :] = self.qoi_fn(X[i, :].reshape(1, -1))
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

    @property
    def y(self) -> Optional[np.ndarray]:
        return self.Y

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
        qoi_fn: Optional[Callable] = None,
        active_keys: Optional[List[str]] = None,
        seed: Optional[int] = None,
        doe_type: str = 'LHS',
        DOE_type: Optional[str] = None,
    ):
        # params is expected to encapsulate parameter layout and optionally
        # metadata like n_out. Accept qoi_fn and active_keys as explicit
        # constructor args that override any attributes on the params object.
        self.params = params
        self.qoi_fn = qoi_fn
        self.active_keys = active_keys if active_keys is not None else getattr(params, "active_keys", None)
        self.seed = int(seed) if seed is not None else None
        # plotting is not supported by sampler_new_cls; omit plot_solution
        # n_out is not stored on params; detect from the provided `qoi_fn` when needed
        self.n_out = None
        # backward compatibility for constructor kwarg
        # backward-compatible attribute name
        self.doe_type = DOE_type if DOE_type is not None else doe_type
        self.DOE_type = self.doe_type

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

    @property
    def X(self) -> Optional[np.ndarray]:
        if self.X_reference is None:
            return None
        return self.reference_to_physical(self.X_reference)

    @property
    def x(self) -> Optional[np.ndarray]:
        return self.X

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

    def _detect_n_out(self) -> None:
        """Auto-detect number of QoI outputs using the params base point."""
        print("n_out not provided -> calling QoI for automatic determination")
        x_center = self.params.pack(self.params.base)
        # Prefer named dict input when the params processor provides an unpack method
        if self.active_keys is not None or hasattr(self.params, "unpack"):
            params_dict = self.params.unpack(x_center)
            qval = self.qoi_fn(params_dict)
        else:
            qval = self.qoi_fn(x_center)
        print(f"QoIs at test point: {qval}")
        # If QoIs returns a dict, build layout to allow slicing by key
        if isinstance(qval, dict):
            self._build_qoi_layout(qval)
            self.n_out = self.qoi_dim
        else:
            arr = np.asarray(qval)
            self.qoi_dim = int(arr.size) if arr.size is not None else int(np.max(arr.shape))
            self._qoi_layout = None
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

    

    def sample(
        self,
        n_samples: Optional[int] = None,
        N: Optional[int] = None,
        as_additional_samples: bool = False,
        as_additional_points: Optional[bool] = None,
        plot: bool = False,
        batch_computation: bool = False,
        **kwargs,
    ) -> None:
        # resolve backward-compatible args: prefer `n_samples`, accept `N` as alias
        if n_samples is None:
            if N is None:
                # accept legacy names passed via kwargs
                n_samples = kwargs.get('n_samples') or kwargs.get('N')
                if n_samples is None:
                    raise ValueError("n_samples (or legacy N) must be provided")
            else:
                n_samples = N

        if as_additional_points is not None:
            as_additional_samples = as_additional_points
        else:
            # also accept legacy name passed in kwargs
            if not as_additional_samples and 'as_additional_points' in kwargs:
                as_additional_samples = kwargs.get('as_additional_points')

        if self.sampler_doe is None or not as_additional_samples:
            if self.sampler_doe is not None and not as_additional_samples:
                print("Warning: Sampling is being reinitialized. Set as_additional_samples=True to continue sampling.")
            try:
                self.sampler_doe = DOEFactory.create(self.doe_type, self.dim, seed=self.seed)
            except ValueError as e:
                raise ValueError(f"Invalid DOE type: {e}")
            X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=False)
        else:
            X_reference = self.sampler_doe.sample(n_samples, as_additional_samples=True)

        # Convert reference -> physical using params processor
        X = self.reference_to_physical(X_reference)

        if self.n_out is not None:
            Y = np.zeros((len(X_reference), self.n_out))
        else:
            Y = None

        sample_type = "additional samples" if as_additional_samples and self.X_reference is not None else "samples"
        print(f"Start computing {len(X_reference)} {sample_type} in parametric dimension {self.dim} using {self.DOE_type}")

        # Decide batch vs sequential based on params processor mode and caller preference
        can_batch = getattr(self.params, "_mode", None) == "array"
        if batch_computation and can_batch and self.qoi_fn is not None:
            Y = self._evaluate_batch(X)
        else:
            Y = self._evaluate_sequential(X)

        print("... done sampling")

        self._store_results(X_reference, X, Y, as_additional_points)

    def reference_to_physical(self, X_reference: np.ndarray) -> np.ndarray:
        """Use params.processor to convert reference to physical."""
        return self.params.reference_to_physical(X_reference)

    def physical_to_reference(self, X: np.ndarray) -> np.ndarray:
        """Use params.processor to convert physical to reference."""
        return self.params.physical_to_reference(X)

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        if self.qoi_fn is None:
            return None

        # If params.unpack exists, ask it to convert the packed X to the
        # form expected by the QoI. For array-style processors this will be
        # a (M,N) ndarray and can be passed directly to the QoI function.
        if hasattr(self.params, "unpack"):
            unpacked = self.params.unpack(X)
            # array-style unpack -> pass batched array to QoI
            if isinstance(unpacked, np.ndarray):
                try:
                    out = self.qoi_fn(unpacked)
                except TypeError:
                    # Fallback: QoI may expect single-sample arrays; warn and evaluate sequentially
                    warnings.warn(
                        "Batch QoI evaluation failed; falling back to sequential evaluation",
                        UserWarning,
                    )
                    outs = []
                    for i in range(unpacked.shape[0]):
                        oi = self.qoi_fn(unpacked[i])
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

                return np.asarray(out)

            # dict-style unpack not supported in batch mode
            raise NotImplementedError("Batch evaluation not implemented for dict-style QoIs (use sequential evaluation)")

        # No unpacking required; pass X directly
        try:
            return self.qoi_fn(X)
        except TypeError as e:
            raise TypeError(f"QoI evaluation failed: {e}") from e

    def _evaluate_sequential(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.qoi_fn is None:
            return None
        Y = np.zeros((len(X), self.n_out)) if self.n_out is not None else None
        for i in range(len(X)):
            try:
                if self.active_keys is not None:
                    params_dict = {self.active_keys[j]: X[i, j] for j in range(len(self.active_keys))}
                    out = self.qoi_fn(params_dict)
                elif hasattr(self.params, 'unpack'):
                    unpacked = self.params.unpack(X[i])
                    # If unpack returns an ndarray (array-style processor), pass
                    # the flat array to the QoI; otherwise assume dict and pass it.
                    if isinstance(unpacked, np.ndarray):
                        to_call = unpacked if unpacked.ndim == 1 else unpacked.reshape(unpacked.shape)
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
            if Y is not None and self.Y is not None:
                self.Y = np.concatenate((self.Y, Y), axis=0)

    @property
    def y(self) -> Optional[np.ndarray]:
        return self.Y
    
