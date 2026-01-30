"""
Test neural network gradient computation against finite differences.

IMPORTANT - PyTorch Installation on Windows:
--------------------------------------------
If these tests are SKIPPED, PyTorch is not properly installed. Common issue on Windows:
"ImportError: DLL load failed while importing _C"

Quick Fix:
1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Then install PyTorch:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Or use Conda (more reliable on Windows):
   conda install -c pytorch pytorch cpuonly -y

See DOCS/SURROPTIM_CHANGELOG_AND_PYTORCH_WINDOWS.md for detailed instructions.

Test Purpose:
-------------
These tests validate that the vectorized gradient computation in neural_network_meta_model.py
correctly computes derivatives using PyTorch autograd. The gradients are compared against
finite difference approximations to ensure mathematical correctness.
"""

import numpy as np
import pytest

try:
    from surroptim.neural_network_meta_model import neural_net_regressor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_neural_net_gradients_vs_finite_difference():
    """Test that neural network gradients match finite difference approximation."""
    rng = np.random.default_rng(42)
    
    # Create simple training data
    X_train = rng.uniform(-1.0, 1.0, size=(50, 3))
    # Simple quadratic function for testing
    y_train = np.sum(X_train**2, axis=1, keepdims=True)
    
    # Train a small network
    model = neural_net_regressor(n_hidden=20)
    model.train(X_train, y_train, lr=1e-2, epochs=500)
    
    # Test points
    X_test = rng.uniform(-1.0, 1.0, size=(5, 3))
    
    # Get predictions and gradients from model
    y_pred, grads_model = model.predict_and_grad(X_test)
    
    # Compute gradients via finite differences
    eps = 1e-5
    N, n_in = X_test.shape
    n_out = y_pred.shape[1]
    grads_fd = np.zeros((N, n_out, n_in))
    
    for i in range(N):
        for j in range(n_in):
            X_plus = X_test.copy()
            X_minus = X_test.copy()
            X_plus[i, j] += eps
            X_minus[i, j] -= eps
            
            y_plus = model.predict(X_plus)
            y_minus = model.predict(X_minus)
            
            grads_fd[i, :, j] = (y_plus[i] - y_minus[i]) / (2 * eps)
    
    # Compare gradients
    max_abs_error = np.max(np.abs(grads_model - grads_fd))
    mean_abs_error = np.mean(np.abs(grads_model - grads_fd))
    relative_error = np.mean(np.abs(grads_model - grads_fd) / (np.abs(grads_fd) + 1e-10))
    
    print(f"Gradient comparison:")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Mean absolute error: {mean_abs_error:.2e}")
    print(f"  Mean relative error: {relative_error:.2e}")
    
    # Check that gradients match within tolerance
    assert max_abs_error < 1e-4, f"Max gradient error too large: {max_abs_error:.2e}"
    assert mean_abs_error < 1e-5, f"Mean gradient error too large: {mean_abs_error:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_neural_net_gradients_multi_output():
    """Test gradients for multi-output neural network."""
    rng = np.random.default_rng(43)
    
    # Create training data with multiple outputs
    X_train = rng.uniform(-1.0, 1.0, size=(100, 2))
    y_train = np.column_stack([
        X_train[:, 0]**2 + X_train[:, 1],
        X_train[:, 0] * X_train[:, 1],
        np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
    ])
    
    # Train network
    model = neural_net_regressor(n_hidden=30)
    model.train(X_train, y_train, lr=1e-2, epochs=1000)
    
    # Test points
    X_test = rng.uniform(-1.0, 1.0, size=(3, 2))
    
    # Get predictions and gradients
    y_pred, grads_model = model.predict_and_grad(X_test)
    
    # Compute finite difference gradients
    eps = 1e-5
    N, n_in = X_test.shape
    n_out = y_pred.shape[1]
    grads_fd = np.zeros((N, n_out, n_in))
    
    for i in range(N):
        for j in range(n_in):
            X_plus = X_test.copy()
            X_minus = X_test.copy()
            X_plus[i, j] += eps
            X_minus[i, j] -= eps
            
            y_plus = model.predict(X_plus)
            y_minus = model.predict(X_minus)
            
            grads_fd[i, :, j] = (y_plus[i] - y_minus[i]) / (2 * eps)
    
    # Check shape
    assert grads_model.shape == (N, n_out, n_in), f"Gradient shape mismatch: {grads_model.shape}"
    
    # Compare gradients
    max_abs_error = np.max(np.abs(grads_model - grads_fd))
    mean_abs_error = np.mean(np.abs(grads_model - grads_fd))
    
    print(f"Multi-output gradient comparison:")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Mean absolute error: {mean_abs_error:.2e}")
    
    # Check that gradients match
    assert max_abs_error < 1e-4, f"Max gradient error too large: {max_abs_error:.2e}"
    assert mean_abs_error < 1e-5, f"Mean gradient error too large: {mean_abs_error:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_neural_net_gradients_single_sample():
    """Test gradients for a single sample (edge case)."""
    rng = np.random.default_rng(44)
    
    # Train on simple data
    X_train = rng.uniform(-1.0, 1.0, size=(30, 4))
    y_train = np.sum(X_train, axis=1, keepdims=True)
    
    model = neural_net_regressor(n_hidden=15)
    model.train(X_train, y_train, lr=1e-2, epochs=500)
    
    # Single test point
    X_test = rng.uniform(-1.0, 1.0, size=(1, 4))
    
    y_pred, grads_model = model.predict_and_grad(X_test)
    
    # Finite differences
    eps = 1e-5
    n_in = X_test.shape[1]
    n_out = y_pred.shape[1]
    grads_fd = np.zeros((1, n_out, n_in))
    
    for j in range(n_in):
        X_plus = X_test.copy()
        X_minus = X_test.copy()
        X_plus[0, j] += eps
        X_minus[0, j] -= eps
        
        y_plus = model.predict(X_plus)
        y_minus = model.predict(X_minus)
        
        grads_fd[0, :, j] = (y_plus[0] - y_minus[0]) / (2 * eps)
    
    max_error = np.max(np.abs(grads_model - grads_fd))
    print(f"Single sample gradient error: {max_error:.2e}")
    
    assert max_error < 1e-4, f"Gradient error too large: {max_error:.2e}"
