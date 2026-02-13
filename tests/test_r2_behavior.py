import numpy as np
from surroptim.util import r2_score, r2_score_simple
from sklearn.metrics import r2_score as sk_r2
from surroptim.meta_models import metamodel


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


def test_metamodel_test_accepts_single_sample_tensor_y_test():
    class _Dummy(metamodel):
        def train(self) -> None:
            return

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # predict zeros with the trained output width
            return np.zeros((X.shape[0], int(self.n_out)))

    rng = np.random.default_rng(1234)
    m = _Dummy()

    # Train with per-sample tensor outputs (N,2,2) -> internal (N,4)
    X_train = rng.normal(size=(5, 3))
    y_train = rng.normal(size=(5, 2, 2))
    m.train_init(X_train, y_train)

    # Test with a *single* sample, but y_test provided as (2,2) (missing sample axis)
    X_test = rng.normal(size=(1, 3))
    y_test = rng.normal(size=(2, 2))

    # Should not raise; metamodel.test should normalize y_test to (1,4)
    m.test(X_test=X_test, y_test=y_test)
