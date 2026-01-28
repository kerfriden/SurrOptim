import numpy as np
from surroptim.doe_strategies import DOEFactory


def test_lhs_2d_reference_space():
    doe = DOEFactory.create('LHS', 2, seed=42)
    N = 16
    X = doe.sample(N)
    assert X.shape == (N, 2)
    assert np.all(X >= -1.0) and np.all(X <= 1.0)

    # request additional points and ensure internal state grows
    X_more = doe.sample(4, as_additional_points=True)
    assert X_more.shape == (4, 2)
    assert doe.X.shape == (N + 4, 2)
    assert np.all(doe.X >= -1.0) and np.all(doe.X <= 1.0)


def test_prs_2d_reference_space():
    doe = DOEFactory.create('PRS', 2, seed=1)
    N = 5
    X = doe.sample(N)
    assert X.shape == (N, 2)
    assert np.all(X >= -1.0) and np.all(X <= 1.0)
import numpy as np
from surroptim.doe_strategies import DOEFactory


def test_lhs_2d_reference_space():
    doe = DOEFactory.create('LHS', 2, seed=42)
    N = 16
    X = doe.sample(N)
    assert X.shape == (N, 2)
    assert np.all(X >= -1.0) and np.all(X <= 1.0)

    # request additional points and ensure internal state grows
    X_more = doe.sample(4, as_additional_points=True)
    assert X_more.shape == (4, 2)
    assert doe.X.shape == (N + 4, 2)
    assert np.all(doe.X >= -1.0) and np.all(doe.X <= 1.0)


def test_prs_2d_reference_space():
    doe = DOEFactory.create('PRS', 2, seed=1)
    N = 5
    X = doe.sample(N)
    assert X.shape == (N, 2)
    assert np.all(X >= -1.0) and np.all(X <= 1.0)
