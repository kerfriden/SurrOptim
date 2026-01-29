import itertools
import numpy as np


def generate_multi_index(D: int, N: int):
    """Generate all multi-indices in dimension D with total order ≤ N."""
    ranges = [range(N + 1) for _ in range(D)]
    return [idx for idx in itertools.product(*ranges) if sum(idx) <= N]


def generate_tensor_product_index(D: int, N: int):
    """Generate tensor-product multi-indices in dimension D with orders 0..N.

    Returns all combinations of orders (i1,...,iD) where each 0 <= ik <= N.
    This produces the full tensor-product / full-factorial polynomial index set.
    """
    if D <= 0:
        raise ValueError(f"Dimension must be positive, got {D}")
    if N < 0:
        raise ValueError(f"Order must be non-negative, got {N}")

    ranges = [range(N + 1) for _ in range(D)]
    return [idx for idx in itertools.product(*ranges)]


def monomials(x: np.ndarray, order: int) -> np.ndarray:
    """Return x**order (elementwise)."""
    return np.asarray(x) ** order


def Legendre(x: np.ndarray, order: int) -> np.ndarray:
    """Legendre polynomial P_order(x) via stable iterative recurrence."""
    x = np.asarray(x)

    if order == 0:
        return np.ones_like(x)
    if order == 1:
        return x

    Pn_2 = np.ones_like(x)  # P0
    Pn_1 = x  # P1

    for n in range(2, order + 1):
        Pn = ((2 * n - 1) * x * Pn_1 - (n - 1) * Pn_2) / n
        Pn_2, Pn_1 = Pn_1, Pn

    return Pn_1


def Legendre_zero_one(x: np.ndarray, order: int) -> np.ndarray:
    """Shifted Legendre polynomial orthogonal over [0,1]."""
    return Legendre(2.0 * np.asarray(x) - 1.0, order)


def poly_basis_multi_index(x: np.ndarray, monomials_1D, MI):
    """Evaluate multivariate monomials given a multi-index list MI.

    Args:
        x: (n_samples, dim) input points.
        monomials_1D: callable (x, order) -> x**order elementwise.
        MI: list of tuples of orders, length = n_basis.

    Returns:
        poly_val: (n_samples, n_basis) array of basis evaluations.
    """
    x = np.asarray(x)
    poly_val = np.ones((len(x), len(MI)))
    for j, orderj in enumerate(MI):
        order_arr = np.asarray(orderj)
        mask = order_arr != 0
        if mask.any():
            powers = order_arr[mask]
            x_selected = x[:, mask]
            monomial_values = np.array([
                monomials_1D(x_selected[:, k], order=powers[k])
                for k in range(len(powers))
            ]).T
            poly_val[:, j] = np.prod(monomial_values, axis=1)
    return poly_val
