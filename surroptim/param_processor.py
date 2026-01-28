import numpy as np
from math import erf, sqrt

# ============================================================
# Normal CDF / inverse CDF (PPF)
#   - CDF uses erf (exact)
#   - PPF prefers erfinv if available; otherwise Acklam approx
# ============================================================

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    np_erf = getattr(np, "erf", None)
    if np_erf is not None:
        return 0.5 * (1.0 + np_erf(x / np.sqrt(2.0)))
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def _norm_ppf_acklam(p):
    """Acklam approximation. p in (0,1)."""
    p = np.asarray(p, dtype=float)

    a = np.array([
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    ])
    b = np.array([
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    ])
    d = np.array([
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    ])

    plow, phigh = 0.02425, 1.0 - 0.02425
    x = np.empty_like(p)

    m = p < plow
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(p[m]))
        x[m] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    m = (p >= plow) & (p <= phigh)
    if np.any(m):
        q = p[m] - 0.5
        r = q*q
        x[m] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    m = p > phigh
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(1.0 - p[m]))
        x[m] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    return x

def norm_ppf(p):
    """
    Inverse CDF of standard normal.
    Uses np.erfinv when available (typically more accurate),
    otherwise falls back to Acklam approximation.
    """
    p = np.asarray(p, dtype=float)
    if hasattr(np, "erfinv"):
        # Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        return np.sqrt(2.0) * np.erfinv(2.0 * p - 1.0)
    return _norm_ppf_acklam(p)

# ============================================================
# Active specs helpers
# ============================================================

def _default_midpoint(lower, upper, distribution):
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if distribution == "linear":
        return 0.5 * (lo + hi)
    if distribution == "log":
        if np.any(lo <= 0) or np.any(hi <= 0):
            raise ValueError("log distribution requires lower, upper > 0")
        return np.sqrt(lo * hi)
    raise ValueError(f"Unknown distribution: {distribution!r}")

def _mask_from_select(select, shape):
    if shape == ():  # scalar
        if select is None:
            return np.array(True, dtype=bool)
        if isinstance(select, (bool, np.bool_)):
            return np.array(bool(select), dtype=bool)
        if isinstance(select, np.ndarray) and select.dtype == bool and select.shape == (): 
            return select.copy()
        raise ValueError("For scalar params, select must be None or a scalar bool.")

    if select is None:
        return np.ones(shape, dtype=bool)
    if isinstance(select, np.ndarray) and select.dtype == bool:
        if select.shape != shape:
            raise ValueError(f"Boolean select shape {select.shape} != param shape {shape}")
        return select.copy()

    m = np.zeros(shape, dtype=bool)
    m[select] = True
    return m

def _trim_selects_last_wins(active_specs, base_params, *, verbose=False, log=print, drop_empty=True):
    """
    If selects overlap for the same param, earlier selects are trimmed so later specs win.
    Returns a NEW dict; all active selects become boolean masks.
    """
    specs = {k: (v if v is False else dict(v)) for k, v in active_specs.items()}
    taken = {}

    for var_id in reversed(list(specs.keys())):
        spec = specs[var_id]
        if spec is False:
            continue

        p = spec.get("param", var_id)
        if p not in base_params:
            raise KeyError(f"Param '{p}' referenced by '{var_id}' not found in init params.")
        shape = np.asarray(base_params[p]).shape

        m = _mask_from_select(spec.get("select", None), shape)
        t = taken.setdefault(p, np.zeros(shape, dtype=bool))

        overlap = m & t
        if np.any(overlap) and verbose:
            log(f"[last-wins] trimming '{var_id}' ({p}): removed {int(overlap.sum())} overlapping entries")

        m = m & ~t

        if drop_empty:
            empty = (not bool(m)) if shape == () else (m.sum() == 0)
            if empty:
                if verbose:
                    log(f"[last-wins] dropping '{var_id}' ({p}): empty after trimming")
                del specs[var_id]
                continue

        spec["param"] = p
        spec["select"] = m
        taken[p] |= m

    return specs

# ============================================================
# ParameterProcessor
# ============================================================

class ParameterProcessor:
    """Unified parameter processor.

    Behavior adapts based on the type of `init_params` passed to the
    constructor:
      - dict -> dict-oriented processor (original behavior)
      - np.ndarray -> array-oriented processor: the array is treated as a
        single base parameter `__arr` and unspecified specs default to it.

    Public API mirrors the previous processors: `pack`, `unpack`,
    `physical_to_reference` / `reference_to_physical`, gaussian helpers, and
    `dim`.
    """

    def __init__(self, init_params, active_specs: dict, *, verbose=False, log=print, eps=1e-12):
        self.verbose = verbose
        self.log = log
        self.eps = float(eps)

            # Placeholder for future implementation
        if isinstance(init_params, np.ndarray):
            self._mode = "array"
            base = {"__arr": init_params.copy()}
            # rewrite specs to target __arr by default
            specs0 = {k: (v if v is False else dict(v)) for k, v in active_specs.items()}
            for var_id, spec in specs0.items():
                if spec is False:
                    continue
                spec.setdefault("param", "__arr")
                if "scale" in spec and "distribution" not in spec:
                    spec["distribution"] = spec["scale"]
        elif isinstance(init_params, dict):
            self._mode = "dict"
            base = {k: np.asarray(v).copy() for k, v in init_params.items()}
            specs0 = {k: (v if v is False else dict(v)) for k, v in active_specs.items()}
            # normalize legacy keys
            for var_id, spec in specs0.items():
                if spec is False:
                    continue
                if "scale" in spec and "distribution" not in spec:
                    spec["distribution"] = spec["scale"]
                spec.setdefault("select", None)
        else:
            raise TypeError("init_params must be a dict or numpy.ndarray")

        # remember scalar keys (for dict-mode only)
        if self._mode == "dict":
            self._scalar_keys = {k for k, v in base.items() if np.asarray(v).shape == ()}
        else:
            self._scalar_keys = {"__arr": False}

        # ensure missing base params are filled using spec midpoints (dict-mode)
        if self._mode == "dict":
            for var_id, spec in specs0.items():
                if spec is False:
                    continue
                p = spec.get("param", var_id)
                spec["param"] = p
                spec.setdefault("distribution", "linear")
                spec.setdefault("select", None)

                if p not in base:
                    sel = spec["select"]
                    if isinstance(sel, np.ndarray) and sel.dtype == bool:
                        shape = sel.shape
                    elif sel is None:
                        shape = ()
                        self._scalar_keys.add(p)
                    else:
                        raise KeyError(f"Param '{p}' missing from init_params; cannot infer shape from select.")
                    fill = _default_midpoint(spec["lower"], spec["upper"], spec["distribution"])
                    base[p] = np.broadcast_to(np.asarray(fill), shape).copy()

        # resolve overlaps and build layout
        self.base = base
        self.specs = _trim_selects_last_wins(specs0, self.base, verbose=verbose, log=log, drop_empty=True)

        self._layout = []
        cursor = 0
        for var_id, spec in self.specs.items():
            if spec is False:
                continue
            p = spec["param"]
            arr = self.base[p]
            mask = spec["select"]
            distribution = spec.get("distribution", "linear")

            n = 1 if np.asarray(arr).shape == () else int(mask.sum())
            sl = slice(cursor, cursor + n)
            cursor += n

            if "lower" not in spec or "upper" not in spec:
                raise ValueError(f"Active spec '{var_id}' must include 'lower' and 'upper'.")

            lo = np.broadcast_to(np.asarray(spec["lower"], dtype=float), (n,)).ravel()
            hi = np.broadcast_to(np.asarray(spec["upper"], dtype=float), (n,)).ravel()
            if np.any(hi <= lo):
                raise ValueError(f"'{var_id}': require upper > lower elementwise.")

            if distribution == "log":
                if np.any(lo <= 0) or np.any(hi <= 0):
                    raise ValueError(f"'{var_id}': log distribution requires > 0.")
                loglo, loghi = np.log(lo), np.log(hi)
            elif distribution == "linear":
                loglo = loghi = None
            else:
                raise ValueError(f"'{var_id}': unknown distribution {distribution!r}")

            self._layout.append({
                "var_id": var_id, "param": p, "mask": mask, "n": n, "sl": sl,
                "distribution": distribution, "lo": lo, "hi": hi, "loglo": loglo, "loghi": loghi
            })

        self.dim = cursor

    def _as_X(self, x_or_X):
        """Accept (N,), (M,N), list-of-(N,) -> returns (X, is_batch)."""
        if isinstance(x_or_X, list):
            X = np.asarray(x_or_X, dtype=float)
        else:
            X = np.asarray(x_or_X, dtype=float)

        if X.ndim == 1:
            if X.size != self.dim:
                raise ValueError(f"Expected length {self.dim}, got {X.size}")
            return X[None, :], False
        if X.ndim == 2:
            if X.shape[1] != self.dim:
                raise ValueError(f"Expected shape (M,{self.dim}), got {X.shape}")
            return X, True
        raise ValueError(f"Expected 1D or 2D array (or list), got ndim={X.ndim}")

    # ---- dict <-> packed physical ----
    def pack(self, params_or_list):
        """Pack parameters into physical packed vector(s).

        - In dict-mode: accept dict -> (N,) or list[dict] -> (M,N).
        - In array-mode: accept ndarray (N,) or (M,N) and validate shape.
        """
        # Dict input (works for both modes when caller provides full base dict)
        if isinstance(params_or_list, dict):
            params = params_or_list
            params_np = {k: np.asarray(v) for k, v in params.items()}
            x = np.zeros(self.dim, dtype=float)

            for it in self._layout:
                p, mask, sl = it["param"], it["mask"], it["sl"]
                arr = np.asarray(params_np.get(p, self.base[p]))
                if np.asarray(arr).shape == ():
                    x[sl] = float(arr)
                else:
                    x[sl] = np.asarray(arr[mask], dtype=float).ravel()
            return x

        # Array-like input (array-mode expected)
        if isinstance(params_or_list, list):
            X = np.asarray(params_or_list, dtype=float)
        else:
            X = np.asarray(params_or_list, dtype=float)

        if X.ndim == 1:
            if X.size != self.dim:
                raise ValueError(f"Expected length {self.dim}, got {X.size}")
            return X.copy()
        if X.ndim == 2:
            if X.shape[1] != self.dim:
                raise ValueError(f"Expected shape (M,{self.dim}), got {X.shape}")
            return X.copy()
        raise ValueError("Expected dict or 1D/2D array (or list)")

    def unpack(self, x_or_X):
        """Unpack packed physical vector(s) to dict(s) or ndarray(s).

        - In array-mode: return ndarray (N,) or (M,N).
        - In dict-mode: return dict or list[dict] as before.
        """
        if self._mode == "array":
            X, is_batch = self._as_X(x_or_X)
            return X.copy() if is_batch else X[0].copy()

        # dict-mode behavior (unchanged)
        X, is_batch = self._as_X(x_or_X)
        outs = []
        for i in range(X.shape[0]):
            x = X[i]
            out = {k: v.copy() for k, v in self.base.items()}
            for it in self._layout:
                p, mask, sl = it["param"], it["mask"], it["sl"]
                arr = out[p]
                if np.asarray(arr).shape == ():
                    out[p] = float(x[sl][0])
                else:
                    a = np.asarray(arr).copy()
                    a[mask] = x[sl]
                    out[p] = a

            # cast scalar keys to python floats for nicer dicts
            for k in self._scalar_keys:
                if k in out and np.asarray(out[k]).shape == ():
                    out[k] = float(out[k])

            outs.append(out)

        return outs if is_batch else outs[0]

    # ---- physical <-> unit [-1,1]^N ----
    def physical_to_unit(self, x_or_X, *, clip=False):
        """Alias name kept for compatibility: physical -> reference (-1,1).

        Use `physical_to_reference` terminology where possible. If `clip=True`,
        values are clipped to the declared bounds prior to mapping.
        """
        X, is_batch = self._as_X(x_or_X)
        Z = np.zeros_like(X)

        for it in self._layout:
            sl, lo, hi, distribution = it["sl"], it["lo"], it["hi"], it["distribution"]
            Y = X[:, sl]
            if clip:
                Y = np.clip(Y, lo, hi)

            if distribution == "linear":
                t = (Y - lo) / (hi - lo)
            else:  # log
                if np.any(Y <= 0):
                    raise ValueError(f"'{it['var_id']}': log distribution requires values > 0")
                t = (np.log(Y) - it["loglo"]) / (it["loghi"] - it["loglo"])

            Z[:, sl] = 2.0 * t - 1.0

        return Z if is_batch else Z[0]

    def physical_to_reference(self, x_or_X, *, clip=False):
        """Map physical parameters to reference space in [-1, 1].

        If `clip=True` values are clipped to the active bounds before mapping.
        """
        X, is_batch = self._as_X(x_or_X)
        Z = np.zeros_like(X)

        for it in self._layout:
            sl, lo, hi, distribution = it["sl"], it["lo"], it["hi"], it["distribution"]
            Y = X[:, sl]
            if clip:
                Y = np.clip(Y, lo, hi)

            if distribution == "linear":
                t = (Y - lo) / (hi - lo)
            else:  # log
                if np.any(Y <= 0):
                    raise ValueError(f"'{it['var_id']}': log distribution requires values > 0")
                t = (np.log(Y) - it["loglo"]) / (it["loghi"] - it["loglo"])

            Z[:, sl] = 2.0 * t - 1.0

        return Z if is_batch else Z[0]

    def reference_to_physical(self, z_or_Z, *, clip=False):
        """Map reference values in [-1, 1] back to physical parameter values.

        Raises ValueError if input is outside [-1,1] unless `clip=True` is used
        on the upstream `physical_to_reference` call.
        """
        Z, is_batch = self._as_X(z_or_Z)
        X = np.zeros_like(Z)

        for it in self._layout:
            sl, lo, hi, distribution = it["sl"], it["lo"], it["hi"], it["distribution"]
            t = 0.5 * (Z[:, sl] + 1.0)
            if clip:
                t = np.clip(t, 0.0, 1.0)

            if distribution == "linear":
                Y = lo + t * (hi - lo)
            else:  # log
                Y = np.exp(it["loglo"] + t * (it["loghi"] - it["loglo"]))

            X[:, sl] = Y

        return X if is_batch else X[0]

    # (removed compatibility aliases)

    def pack_unit(self, params_or_list, *, clip=False):
        """Compatibility: pack physical dict to reference vector (-1,1)."""
        return self.physical_to_unit(self.pack(params_or_list), clip=clip)

    def unpack_unit(self, z_or_Z, *, clip=False):
        """Compatibility: unpack reference vector (-1,1) to physical dict."""
        return self.unpack(self.reference_to_physical(z_or_Z, clip=clip))

    # ---- unit <-> gaussian ----
    def unit_to_gauss(self, z_or_Z, *, eps=None):
        """Convert reference Z in [-1,1] to gaussian space via inverse CDF.

        Raises ValueError if input reference values lie outside [-1,1].
        """
        Z, is_batch = self._as_X(z_or_Z)
        # Validate reference range
        if np.any(Z < -1.0) or np.any(Z > 1.0):
            raise ValueError("Reference values must be within [-1, 1].")
        eps = self.eps if eps is None else float(eps)

        U = 0.5 * (Z + 1.0)
        U = np.clip(U, eps, 1.0 - eps)
        G = norm_ppf(U)

        return G if is_batch else G[0]

    def gauss_to_unit(self, g_or_G):
        """Convert gaussian values to reference Z in [-1,1]."""
        G, is_batch = self._as_X(g_or_G)
        U = norm_cdf(G)
        Z = 2.0 * U - 1.0
        return Z if is_batch else Z[0]

    # Backward-compatible alias: gaussian -> reference
    def gaussian_to_reference(self, g_or_G):
        """Alias for `gauss_to_unit` for callers expecting `gaussian_to_reference`."""
        return self.gauss_to_unit(g_or_G)

    # ---- direct physical <-> gaussian ----
    def physical_to_gauss(self, x_or_X, *, clip=False, eps=None):
        return self.unit_to_gauss(self.physical_to_reference(x_or_X, clip=clip), eps=eps)

    def gauss_to_physical(self, g_or_G, *, clip=False):
        return self.reference_to_physical(self.gauss_to_unit(g_or_G), clip=clip)

    # ---- convenience: reference -> dict and gauss -> dict ----
    def unit_to_dict(self, z_or_Z, *, clip=False):
        """Compatibility wrapper: convert reference vector to physical dict."""
        return self.unpack_unit(z_or_Z, clip=clip)

    def gauss_to_dict(self, g_or_G, *, clip=False):
        return self.unpack(self.gauss_to_physical(g_or_G, clip=clip))

    # New convenience shortcuts: gaussian -> physical (array or dict)
    def gaussian_to_physical(self, g_or_G, *, clip=False):
        """Convenience: convert gaussian values directly to physical packed array.

        Equivalent to `reference_to_physical(gauss_to_unit(g))` but provided
        as a single, discoverable method for convenience.
        """
        return self.gauss_to_physical(g_or_G, clip=clip)

    def gaussian_to_physical_dict(self, g_or_G, *, clip=False):
        """Convenience: convert gaussian values directly to an unpacked physical dict."""
        return self.gauss_to_dict(g_or_G, clip=clip)

    # (removed compatibility aliases)

    # ---- parameter slice/index helpers (dict-mode) ----
    def parameter_slices(self, keys):
        """Return slice(s) into the packed vector for parameter key(s).

        If `keys` is a string, returns a single `slice` for that key. If
        `keys` is a sequence, returns a dict mapping each key to its slice.
        Raises KeyError if the layout is unavailable or a key is missing.
        """
        if not hasattr(self, "_layout") or self._layout is None:
            raise KeyError("No parameter layout available")

        def _single(k: str) -> slice:
            for item in self._layout:
                if item.get("var_id") == k:
                    return item["sl"]
            raise KeyError(f"Parameter '{k}' not found in layout")

        if isinstance(keys, str):
            return _single(keys)
        return {k: _single(k) for k in keys}

    def parameter_indices(self, keys):
        """Return integer index array(s) for parameter key(s).

        If `keys` is a string, returns a 1D `np.ndarray` of indices for that
        key. If `keys` is a sequence, returns a concatenated 1D `np.ndarray`
        of indices in the order of `keys` provided.
        """
        if isinstance(keys, str):
            sl = self.parameter_slices(keys)
            return np.arange(sl.start, sl.stop)

        # keys is sequence
        mapping = self.parameter_slices(keys)
        parts = [np.arange(mapping[k].start, mapping[k].stop) for k in keys]
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)



# ============================================================
# Option A: Params class (stores init_params + active_specs)
# ============================================================

class params_cls(ParameterProcessor):
    """
    Constructor-style params helper.

    Preferred usage: instantiate with `params_cls(init_params=..., active_specs=...)`.
    Backwards-compatible with previous `Params_cls` subclassing pattern.
    """
    def __init__(self, init_params: dict | None = None, active_specs: dict | None = None, *, verbose=False, log=print, eps=1e-12):
        # Allow legacy subclasses to set params0/active_specs_input (handled by passing None)
        if init_params is None:
            init_params = getattr(self, "params0", None)
        if active_specs is None:
            active_specs = getattr(self, "active_specs_input", None)
            if active_specs is None:
                active_specs = getattr(self, "active_specs", None)

        if init_params is None or active_specs is None:
            raise ValueError("Provide init_params/active_specs or set self.params0 and self.active_specs_input in the subclass before super().__init__()")

        self.init_params = init_params
        self.active_specs_input = active_specs
        super().__init__(init_params, active_specs, verbose=verbose, log=log, eps=eps)

    @property
    def active_specs(self):
        """Resolved specs (after last-wins trimming)."""
        return self.specs

    def describe_layout(self):
        lines = [f"dim = {self.dim}"]
        for it in self._layout:
            sl = it["sl"]
            lines.append(
                f"  {it['var_id']:>10s}: param='{it['param']}', n={it['n']}, "
                f"slice=[{sl.start}:{sl.stop}], distribution={it['distribution']}"
            )
        return "\n".join(lines)


# Note: backward-compatible top-level aliases removed
