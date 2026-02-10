import numpy as np

from surroptim.polynomial_meta_models import polynomial_lasso_regressor, polynomial_ridge_regressor


def test_polynomial_SG_infers_level_and_matches_point_count():
    # Build an SG design and ensure the polynomial SG basis matches its size
    from surroptim.sparse_grid import generate_sparse_grid

    dim = 2
    level = 4
    X = generate_sparse_grid(dim, level)
    # simple scalar output
    y = np.sum(X, axis=1, keepdims=True)

    model = polynomial_ridge_regressor(SG=True, level=None)
    A = model.train_init(X, y)

    assert model.level == level
    assert A.shape[0] == X.shape[0]
    assert A.shape[1] == X.shape[0]
from surroptim.util import r2_score


def sigmoid_qoi(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x, y = X[:, 0], X[:, 1]
    s = x + y**2
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


def test_polynomial_ridge_sigmoid_qoi():
    rng = np.random.default_rng(42)
    # Train data in [-2,2]^2
    X_train = rng.uniform(-2.0, 2.0, size=(80, 2))
    y_train = sigmoid_qoi(X_train)

    # Test data
    X_test = rng.uniform(-2.0, 2.0, size=(40, 2))
    y_test = sigmoid_qoi(X_test)

    # Polynomial ridge regressor of order 4 to capture curvature
    model = polynomial_ridge_regressor(order=4, coeff_reg=1e-6, SG=False)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.9, f"R2 too low: {r2:.3f}"


def test_polynomial_lasso_fallback_fista():
    rng = np.random.default_rng(7)
    # Sparse linear relation with bias; lasso should recover via NumPy fallback
    X_train = rng.normal(size=(120, 2))
    noise = 0.01 * rng.normal(size=120)
    y_train = (1.5 * X_train[:, 0] - 0.5 * X_train[:, 1] + 0.2 + noise).reshape(-1, 1)

    X_test = rng.normal(size=(40, 2))
    y_test = (1.5 * X_test[:, 0] - 0.5 * X_test[:, 1] + 0.2).reshape(-1, 1)

    model = polynomial_lasso_regressor(order=1, coeff_reg=1e-3, SG=False, use_sklearn=False)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.98, f"FISTA fallback R2 too low: {r2:.3f}"
