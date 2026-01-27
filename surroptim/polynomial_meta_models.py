import warnings
import numpy as np
try:
    from sklearn.linear_model import Lasso as SklearnLasso  # optional
except ImportError:
    SklearnLasso = None

from surroptim.meta_models import metamodel
from surroptim.polynomials import monomials, generate_multi_index, poly_basis_multi_index
from surroptim.sparse_grid import generate_list_orders_dim


class polynomial_regressor(metamodel):
    """Base polynomial regressor with configurable basis and index set."""

    def __init__(self, order, basis_generator=None, coeff_reg=None, SG: bool = False):
        super().__init__()
        self.order = order
        self.basis_generator = basis_generator or monomials
        self.coeff_reg = coeff_reg if coeff_reg is not None else 1.0e-10
        self.SG = SG
        self.MI = None
        self.weights = None

    def train_init(self, X=None, y=None):
        super().train_init(X, y)
        if self.SG:
            self.MI = generate_list_orders_dim(self.dim, self.order)
        else:
            self.MI = generate_multi_index(self.dim, self.order)
        A = poly_basis_multi_index(X, self.basis_generator, self.MI)
        return A

    def predict(self, X):
        A = poly_basis_multi_index(X, self.basis_generator, self.MI)
        return A @ self.weights

    def train(self, X=None, y=None):
        raise NotImplementedError("train must be implemented in subclasses")


class polynomial_lasso_regressor(polynomial_regressor):
    def __init__(self, order=2, basis_generator=None, coeff_reg=None, SG: bool = False, use_sklearn: bool = True):
        super().__init__(order, basis_generator, coeff_reg, SG)
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
    def __init__(self, order=2, basis_generator=None, coeff_reg=None, SG: bool = False):
        super().__init__(order, basis_generator, coeff_reg, SG)

    def train(self, X=None, y=None):
        A = super().train_init(X, y)
        ATA = A.T @ A
        reg = self.coeff_reg * np.eye(ATA.shape[0])
        self.weights = np.linalg.solve(ATA + reg, A.T @ y)
