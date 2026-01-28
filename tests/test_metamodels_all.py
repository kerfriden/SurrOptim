"""
Tests for all metamodel regressors: GP, neural network, and k-NN.
"""

import numpy as np
import pytest
from surroptim.gaussian_process_meta_model import GP_regressor
from surroptim.neural_network_meta_model import neural_net_regressor, neural_net_regressor_pt
from surroptim.neighrest_neighbour_meta_model import NNeigh_regressor
from surroptim.util import r2_score_simple


def sigmoid_qoi(X: np.ndarray) -> np.ndarray:
    """Sigmoid QoI for testing."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[:, 0]
    y = X[:, 1]
    s = x + y**2
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


def test_gp_regressor_sigmoid_qoi():
    """Test Gaussian Process regressor on sigmoid QoI."""
    rng = np.random.default_rng(42)
    # Train data in [-1,1]^2 (normalized)
    X_train = rng.uniform(-1.0, 1.0, size=(20, 2))
    y_train = sigmoid_qoi(X_train)

    # Test data
    X_test = rng.uniform(-1.0, 1.0, size=(10, 2))
    y_test = sigmoid_qoi(X_test)

    # GP regressor
    model = GP_regressor(length_scale=0.5, length_scale_bounds=(0.1, 2.0))
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score_simple(y_test, y_pred)
    print(f"GP R2 on sigmoid: {r2:.3f}")
    assert r2 > 0.8, f"GP R2 too low: {r2:.3f}"


def test_neural_net_regressor_sigmoid_qoi():
    """Test neural network regressor on sigmoid QoI."""
    rng = np.random.default_rng(43)
    # Train data in [-1,1]^2 (normalized)
    X_train = rng.uniform(-1.0, 1.0, size=(30, 2))
    y_train = sigmoid_qoi(X_train)

    # Test data
    X_test = rng.uniform(-1.0, 1.0, size=(10, 2))
    y_test = sigmoid_qoi(X_test)

    # Neural network regressor
    model = neural_net_regressor(n_hidden=50)
    model.train(X_train, y_train, lr=1e-2, epochs=5000, restart=True)

    y_pred = model.predict(X_test)
    r2 = r2_score_simple(y_test, y_pred)
    print(f"Neural Net R2 on sigmoid: {r2:.3f}")
    assert r2 > 0.7, f"Neural Net R2 too low: {r2:.3f}"


def test_neural_net_regressor_pt_sigmoid_qoi():
    """Test neural network regressor (PyTorch interface) on sigmoid QoI."""
    import torch
    
    rng = np.random.default_rng(44)
    # Train data in [-1,1]^2 (normalized)
    X_train_np = rng.uniform(-1.0, 1.0, size=(30, 2))
    y_train_np = sigmoid_qoi(X_train_np)
    
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()

    # Test data
    X_test_np = rng.uniform(-1.0, 1.0, size=(10, 2))
    y_test_np = sigmoid_qoi(X_test_np)
    X_test = torch.from_numpy(X_test_np).float()

    # Neural network regressor (PyTorch)
    model = neural_net_regressor_pt(n_hidden=50)
    model.train(X_train, y_train, lr=1e-2, epochs=5000, restart=True)

    y_pred = model.predict(X_test).detach().numpy()
    r2 = r2_score_simple(y_test_np, y_pred)
    print(f"Neural Net (PT) R2 on sigmoid: {r2:.3f}")
    assert r2 > 0.7, f"Neural Net (PT) R2 too low: {r2:.3f}"


def test_knn_regressor_sigmoid_qoi():
    """Test k-NN regressor on sigmoid QoI."""
    rng = np.random.default_rng(45)
    # Train data in [-1,1]^2 (normalized)
    X_train = rng.uniform(-1.0, 1.0, size=(30, 2))
    y_train = sigmoid_qoi(X_train)

    # Test data
    X_test = rng.uniform(-1.0, 1.0, size=(10, 2))
    y_test = sigmoid_qoi(X_test)

    # k-NN regressor
    model = NNeigh_regressor(n_neighbors=5, weights='distance')
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score_simple(y_test, y_pred)
    print(f"k-NN R2 on sigmoid: {r2:.3f}")
    assert r2 > 0.5, f"k-NN R2 too low: {r2:.3f}"


def test_gp_with_std():
    """Test GP regressor uncertainty estimates."""
    rng = np.random.default_rng(46)
    X_train = rng.uniform(-1.0, 1.0, size=(15, 2))
    y_train = sigmoid_qoi(X_train)

    X_test = rng.uniform(-1.0, 1.0, size=(5, 2))

    model = GP_regressor(length_scale=1.0)
    model.train(X_train, y_train)

    y_pred, y_std = model.predict(X_test, return_std=True)
    
    assert y_pred.shape[0] == 5
    assert y_std.shape[0] == 5
    assert np.all(y_std >= 0), "Standard deviation should be non-negative"
    print(f"GP predictions shape: {y_pred.shape}, std shape: {y_std.shape}")


def test_knn_with_std_ignored():
    """Test k-NN regressor return_std parameter (should be ignored)."""
    rng = np.random.default_rng(47)
    X_train = rng.uniform(-1.0, 1.0, size=(20, 2))
    y_train = sigmoid_qoi(X_train)

    X_test = rng.uniform(-1.0, 1.0, size=(5, 2))

    model = NNeigh_regressor(n_neighbors=3)
    model.train(X_train, y_train)

    y_pred, y_std = model.predict(X_test, return_std=True)
    
    assert y_pred.shape[0] == 5
    assert y_std is None, "k-NN should return None for std"
    print("k-NN correctly ignores return_std parameter")


if __name__ == "__main__":
    print("=== Testing Gaussian Process Regressor ===")
    test_gp_regressor_sigmoid_qoi()
    print("✓ GP regressor test passed\n")

    print("=== Testing Neural Network Regressor ===")
    test_neural_net_regressor_sigmoid_qoi()
    print("✓ Neural network regressor test passed\n")

    print("=== Testing Neural Network Regressor (PyTorch) ===")
    test_neural_net_regressor_pt_sigmoid_qoi()
    print("✓ Neural network regressor (PT) test passed\n")

    print("=== Testing k-NN Regressor ===")
    test_knn_regressor_sigmoid_qoi()
    print("✓ k-NN regressor test passed\n")

    print("=== Testing GP with Uncertainty ===")
    test_gp_with_std()
    print("✓ GP uncertainty test passed\n")

    print("=== Testing k-NN with return_std ===")
    test_knn_with_std_ignored()
    print("✓ k-NN return_std test passed\n")

    print("All metamodel tests passed!")
