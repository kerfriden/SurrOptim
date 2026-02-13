import numpy as np
from surroptim.util import r2_score, r2_score_simple
from sklearn.metrics import r2_score as sk_r2


def test_r2_matches_sklearn_for_single_output_column():
    rng = np.random.default_rng(123)
    N = 16
    # y_true as (N,1)
    y_true = rng.normal(loc=0.0, scale=1.0, size=(N, 1))
    # y_pred as 1-D (N,)
    y_pred_1d = (y_true.ravel() * 0.8 + 0.1).astype(float)

    # util.r2_score should behave like sklearn (averaged single-output)
    util_r2 = r2_score(y_true, y_pred_1d)
    sk_r2_val = sk_r2(y_true, y_pred_1d)

    assert np.allclose(util_r2, sk_r2_val, atol=1e-12)


def test_r2_simple_matches_flattened():
    rng = np.random.default_rng(456)
    N = 10
    y_true = rng.normal(size=(N, 1))
    y_pred = (y_true.ravel() * 0.5 + 0.2)

    # r2_score_simple should equal sklearn when both are flattened
    assert np.allclose(r2_score_simple(y_true, y_pred), sk_r2(y_true.ravel(), y_pred.ravel()))


def test_r2_flattens_tensor_outputs_per_sample():
    rng = np.random.default_rng(789)
    N = 25
    # Simulate per-sample tensor QoI output (e.g., 2x2 per sample)
    y_true = rng.normal(size=(N, 2, 2))
    y_pred = y_true * 0.9 + 0.05

    # Our r2_score should behave like sklearn on the flattened (N, 4) view
    util_r2 = r2_score(y_true, y_pred)
    sk_r2_val = sk_r2(y_true.reshape(N, -1), y_pred.reshape(N, -1))
    assert np.allclose(util_r2, sk_r2_val, atol=1e-12)
