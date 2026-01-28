"""Backward-compat shim for `qoi` module.

This file keeps the old module path `surroptim.QoI` working by
re-exporting the `QoI_cls` from the canonical `surroptim.qoi` module.
"""

from .qoi import QoI_cls  # noqa: F401
import numpy as np
from typing import Dict, Any, List, Optional, Callable


class QoI_cls:
    """
    Base Quantity-of-Interest helper.

    Two usage patterns are supported:
      - Subclassing and implementing `qoi(self, params_dict) -> dict`.
      - Passing a `qoi_func(params_dict) -> dict` callable at construction.

    Parameters
    ----------
    params : ParameterProcessor | Params
        Parameter handler that already implements pack/unpack/base.
    qoi_func : callable, optional
        If provided, this callable will be used as the QoI evaluator instead of
        requiring a subclass to implement `qoi`.
    """

    def __init__(self, params: Any, qoi_func: Optional[Callable] = None):
        self.params = params

        # If a callable is provided, use it as the qoi implementation.
        if qoi_func is not None:
            if not callable(qoi_func):
                raise TypeError("qoi_func must be callable")
            # bind the callable as an instance method
            self.qoi = lambda params_dict: qoi_func(params_dict)

        # Build a reference layout from a sample QoI evaluation so we can flatten deterministically.
        ref_params = self.params.unpack(self.params.pack(self.params.base))
        ref_qoi = self.qoi(ref_params)
        self._layout = self._build_layout(ref_qoi)
        self.qoi_dim = sum(item["size"] for item in self._layout)

    # ---- to be implemented by subclasses ----
    def qoi(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement qoi(params_dict) in subclass")

    # ---- flattening helpers ----
    def _build_layout(self, qoi_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(qoi_dict, dict):
            raise TypeError("qoi must return a dict")

        layout = []
        for key, val in qoi_dict.items():
            arr = np.asarray(val)
            layout.append({
                "key": key,
                "shape": arr.shape,
                "size": int(arr.size),
            })
        return layout

    def _flatten(self, qoi_dict: Dict[str, Any]) -> np.ndarray:
        parts = []
        for item in self._layout:
            key = item["key"]
            if key not in qoi_dict:
                raise KeyError(f"QoI output missing key '{key}'")
            arr = np.asarray(qoi_dict[key])
            if arr.shape != item["shape"]:
                raise ValueError(
                    f"QoI entry '{key}' shape {arr.shape} != expected {item['shape']}"
                )
            parts.append(arr.reshape(-1))

        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    # ---- public API ----
    def compute_QoI(self, x_flat: np.ndarray) -> np.ndarray:
        """Map flat physical parameters -> flat QoI vector."""
        x_flat = np.asarray(x_flat, dtype=float)
        if x_flat.ndim != 1:
            raise ValueError("compute_QoI expects a 1D array of physical parameters")

        params_dict = self.params.unpack(x_flat)
        qoi_dict = self.qoi(params_dict)
        return self._flatten(qoi_dict)
