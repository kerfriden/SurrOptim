import warnings
import numpy as np
from typing import Optional
try:
    from sklearn.linear_model import Lasso as SklearnLasso  # optional
except ImportError:
    SklearnLasso = None

from surroptim.meta_models import metamodel
from surroptim.polynomials import monomials, generate_multi_index, poly_basis_multi_index
from surroptim.sparse_grid import generate_list_orders_dim


class polynomial_regressor(metamodel):
    """Base polynomial regressor with configurable basis and index set.

        Notes:
                - When `SG=True`, the sparse-grid API expects a hierarchical `level`.
                - If `level` is omitted, the model will try to infer it from the
                    training design size (works when X comes from the library sparse-grid DOE).
                - The `order` argument controls total-degree polynomial order when SG=False.
                    It is not used as a sparse-grid level.
    """

    def __init__(self, order: int = 2, level: Optional[int] = None, basis_generator=None, coeff_reg=None, SG: bool = False, tensor: bool = False):
        super().__init__()
        self.order = order
        self.level = level
        self.basis_generator = basis_generator or monomials
        self.coeff_reg = coeff_reg if coeff_reg is not None else 1.0e-10
        self.SG = SG
        # If True, use full tensor-product multi-index (order per-dimension up to `order`)
        self.tensor = tensor
        self.MI = None
        self.weights = None
        # In SG mode we use `level` (hierarchical refinement) rather than a
        # polynomial total-degree `order`. If level is missing we will attempt
        # to infer it during training from the number of points in X.
        if self.SG and self.level is None:
            # Keep a gentle warning for callers who might be passing `order`
            # expecting sparse-grid behavior.
            if self.order not in (None, 2):
                warnings.warn(
                    "SG=True uses `level=` (hierarchical refinement). The `order` argument is ignored in SG mode.",
                    UserWarning,
                )

    def train_init(self, X=None, y=None):
        super().train_init(X, y)
        if self.SG:
            # generate_list_orders_dim expects the sparse-grid refinement level
            if self.level is None:
                # Infer level from the design size when possible.
                try:
                    from surroptim.sparse_grid import generate_sparse_grid

                    n = int(np.asarray(X).shape[0])
                    inferred = None
                    # brute-force small search; sparse grid level is typically small
                    for lvl in range(1, 1 + 32):
                        try:
                            if len(generate_sparse_grid(self.dim, lvl)) == n:
                                inferred = lvl
                                break
                        except Exception:
                            break
                    if inferred is None:
                        raise ValueError(
                            f"Could not infer sparse-grid level from X (n_points={n}, dim={self.dim}). "
                            "Pass `level=` explicitly."
                        )
                    self.level = int(inferred)
                except Exception as e:
                    raise ValueError(
                        "SG=True requires `level=` unless it can be inferred from the training design size."
                    ) from e
            # Validate that the provided/inferred level matches the training design size.
            try:
                from surroptim.sparse_grid import generate_sparse_grid

                expected_n = len(generate_sparse_grid(self.dim, int(self.level)))
                n = int(np.asarray(X).shape[0])
                if expected_n != n:
                    raise ValueError(
                        f"SG polynomial basis level={self.level} implies {expected_n} sparse-grid points, "
                        f"but got X with {n} points. Use matching `level=` or matching DOE."
                    )
            except ImportError:
                pass

            self.MI = generate_list_orders_dim(self.dim, int(self.level))
        else:
            if getattr(self, 'tensor', False):
                # full tensor-product / full-factorial multi-index
                from surroptim.polynomials import generate_tensor_product_index

                self.MI = generate_tensor_product_index(self.dim, self.order)
            else:
                # total-order multi-index (default)
                self.MI = generate_multi_index(self.dim, self.order)
        A = poly_basis_multi_index(X, self.basis_generator, self.MI)
        return A

    def predict(self, X):
        A = poly_basis_multi_index(X, self.basis_generator, self.MI)
        return A @ self.weights

    def train(self, X=None, y=None):
        raise NotImplementedError("train must be implemented in subclasses")


class polynomial_lasso_regressor(polynomial_regressor):
    def __init__(self, order: int = 2, level: Optional[int] = None, basis_generator=None, coeff_reg=None, SG: bool = False, tensor: bool = False, use_sklearn: bool = True):
        super().__init__(order, level, basis_generator, coeff_reg, SG, tensor)
        self.use_sklearn = use_sklearn

    def _lasso_fista(self, A, y, alpha=1e-3, max_iter=5000, tol=1e-6):
        """Minimal FISTA solver for multi-output Lasso: 0.5||Aw - y||^2 + alpha||w||_1."""
        A = np.asarray(A)
        y = np.asarray(y)
        n, p = A.shape
        # Lipschitz constant of gradient (spectral norm squared)
        L = np.linalg.norm(A, 2) ** 2
        step = 1.0 / (L + 1e-12)

        w = np.zeros((p, y.shape[1]))
        z = w.copy()
        t = 1.0
        At = A.T

        def soft_threshold(zv, lam):
            return np.sign(zv) * np.maximum(np.abs(zv) - lam, 0.0)

        for _ in range(max_iter):
            Aw_minus_y = A @ z - y
            grad = At @ Aw_minus_y
            w_new = soft_threshold(z - step * grad, step * alpha)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = w_new + ((t - 1.0) / t_new) * (w_new - w)
            if np.linalg.norm(w_new - w) < tol * (1.0 + np.linalg.norm(w)):
                w = w_new
                break
            w, t = w_new, t_new
        return w

    def _report_sparsity(self):
        if self.weights is None:
            return
        zero_mask = np.isclose(self.weights, 0.0, atol=1e-12)
        n_zero = int(np.count_nonzero(zero_mask))
        n_total = int(self.weights.size)
        n_nonzero = n_total - n_zero
        zero_frac = n_zero / max(n_total, 1)
        print(f"Lasso weights: {n_zero} zero, {n_nonzero} non-zero (zero fraction {zero_frac:.3f})")
        if zero_frac < 0.10 or zero_frac > 0.90:
            warnings.warn(
                "Lasso sparsity is outside 10%-90% range; consider tuning alpha.",
                RuntimeWarning,
            )

    def train(self, X=None, y=None):
        A = super().train_init(X, y)
        if self.use_sklearn and SklearnLasso is not None:
            model = SklearnLasso(alpha=self.coeff_reg, fit_intercept=False, max_iter=100000)
            model.fit(A, y)
            self.weights = model.coef_.T
        else:
            self.weights = self._lasso_fista(A, y, alpha=self.coeff_reg)

        self._report_sparsity()


class polynomial_ridge_regressor(polynomial_regressor):
    def __init__(self, order: int = 2, level: Optional[int] = None, basis_generator=None, coeff_reg=None, SG: bool = False, tensor: bool = False):
        super().__init__(order, level, basis_generator, coeff_reg, SG, tensor)

    def train(self, X=None, y=None):
        A = super().train_init(X, y)
        ATA = A.T @ A
        reg = self.coeff_reg * np.eye(ATA.shape[0])
        self.weights = np.linalg.solve(ATA + reg, A.T @ y)
