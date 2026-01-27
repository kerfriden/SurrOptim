import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from sampler import sampler_cls
from DOE import DOE_cls
from util import prediction_plot, surf_plot


def test_doe_prs():
    """Test PRS (Pseudo-Random Sampling) DOE"""
    print("\n=== Test DOE: PRS ===")
    doe = DOE_cls(dim=2, DOE_type="PRS")
    X = doe.sample(N=10)
    print(f"PRS samples shape: {X.shape}")
    assert X.shape == (10, 2), "PRS sample shape mismatch"
    assert np.all((X >= -1) & (X <= 1)), "PRS samples should be in [-1,1]"
    print("✓ PRS test passed")


def test_doe_lhs():
    """Test LHS (Latin Hypercube Sampling) DOE"""
    print("\n=== Test DOE: LHS ===")
    doe = DOE_cls(dim=3, DOE_type="LHS")
    X = doe.sample(N=15)
    print(f"LHS samples shape: {X.shape}")
    assert X.shape == (15, 3), "LHS sample shape mismatch"
    assert np.all((X >= -1) & (X <= 1)), "LHS samples should be in [-1,1]"
    print("✓ LHS test passed")


def test_doe_qrs():
    """Test QRS (Quasi-Random Sobol) DOE"""
    print("\n=== Test DOE: QRS ===")
    doe = DOE_cls(dim=2, DOE_type="QRS")
    X = doe.sample(N=16)
    print(f"QRS samples shape: {X.shape}")
    assert X.shape == (16, 2), "QRS sample shape mismatch"
    assert np.all((X >= -1) & (X <= 1)), "QRS samples should be in [-1,1]"
    print("✓ QRS test passed")


def test_doe_sg():
    """Test SG (Sparse Grid) DOE"""
    print("\n=== Test DOE: SG ===")
    doe = DOE_cls(dim=2, DOE_type="SG")
    X = doe.sample(N=3)  # N is refinement level
    print(f"SG samples shape: {X.shape}")
    assert X.shape[1] == 2, "SG dimension mismatch"
    assert np.all((X >= -1) & (X <= 1)), "SG samples should be in [-1,1] after normalization"
    print(f"✓ SG test passed ({X.shape[0]} points)")


def simple_qoi(X):
    """
    Simple QoI function for testing.
    
    Output 1: Simple sum of parameters
    Output 2: Rosenbrock function (valley test functional)
        f(x,y) = (1-x)^2 + 100(y-x^2)^2
    
    This is a classic test function in optimization with a narrow valley.
    Minimum at (1, 1) with f(1,1) = 0.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    n_samples = X.shape[0]
    n_params = X.shape[1]
    
    # Output 1: Simple sum
    sum_output = np.sum(X, axis=1, keepdims=True)
    
    # Output 2: Rosenbrock or simple quadratic for 1D
    if n_params >= 2:
        # 2D Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        x = X[:, 0]
        y = X[:, 1]
        rosenbrock = (1 - x)**2 + 100 * (y - x**2)**2
        output2 = rosenbrock.reshape(-1, 1)
    else:
        # For 1D, use simple quadratic
        output2 = (X[:, 0] - 0.5)**2
        output2 = output2.reshape(-1, 1)
    
    return np.hstack([sum_output, output2])


def simple_qoi_1d(X):
    """1D QoI: sum and quadratic (x - 0.5)^2"""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    output1 = np.sum(X, axis=1, keepdims=True)
    output2 = (X[:, 0] - 0.5)**2
    return np.hstack([output1, output2.reshape(-1, 1)])


def test_sampler_uniform():
    """Test sampler with uniform distribution"""
    print("\n=== Test Sampler: Uniform Distribution ===")
    types = ['uniform', 'uniform']
    params = [[-2, 2], [-2, 2]]
    
    sampler = sampler_cls(
        types=types,
        params=params,
        compute_QoIs=simple_qoi,
        n_out=2
    )
    
    sampler.sampling(N=20, DOE_type='PRS')
    print(f"Samples shape: {sampler.X.shape}")
    print(f"QoI shape: {sampler.Y.shape}")
    assert sampler.X.shape == (20, 2), "Sample shape mismatch"
    assert sampler.Y.shape == (20, 2), "QoI shape mismatch"
    assert np.all(sampler.X[:, 0] >= -2) and np.all(sampler.X[:, 0] <= 2), "X1 out of bounds"
    assert np.all(sampler.X[:, 1] >= -2) and np.all(sampler.X[:, 1] <= 2), "X2 out of bounds"
    
    # Plot the results
    print("Plotting 2D samples colored by first QoI (sum)...")
    prediction_plot(X=sampler.X, y=sampler.Y[:, 0], xlabel='Param 1', ylabel='Param 2', clabel='Sum', show=False)
    
    print("✓ Uniform distribution test passed")


def test_sampler_loguniform():
    """Test sampler with log-uniform distribution"""
    print("\n=== Test Sampler: Log-Uniform Distribution ===")
    types = ['log_uniform']
    params = [[0, 2]]  # log space bounds: [e^0, e^2] = [1, 7.389...]
    
    sampler = sampler_cls(
        types=types,
        params=params,
        compute_QoIs=simple_qoi_1d,
        n_out=2
    )
    
    sampler.sampling(N=20, DOE_type='PRS')
    print(f"Samples shape: {sampler.X.shape}")
    print(f"QoI shape: {sampler.Y.shape}")
    print(f"Sample range: [{np.min(sampler.X):.6f}, {np.max(sampler.X):.6f}]")
    assert sampler.X.shape == (20, 1), "Sample shape mismatch"
    assert np.all(sampler.X > 0), "Log-uniform samples should be positive"
    assert np.all(sampler.X >= np.exp(0)), "Samples below lower bound"
    assert np.all(sampler.X <= np.exp(2)), "Samples above upper bound"
    
    # Plot the results
    print("Plotting 1D log-uniform samples...")
    prediction_plot(X=sampler.X, y=sampler.Y[:, 1], xlabel='Parameter', ylabel='QoI', show=False)
    
    print("✓ Log-uniform distribution test passed")


def test_sampler_sigmoid_qoi():
    """Test sampler with sigmoid(x + y^2) QoI over [-2,2]^2."""
    print("\n=== Test Sampler: Sigmoid QoI ===")

    def sigmoid_qoi(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, 0]
        y = X[:, 1]
        s = x + y
        sig = 1.0 / (1.0 + np.exp(-(x + y**2)))
        return np.vstack([s, sig]).T

    types = ['uniform', 'uniform']
    params = [[-2, 2], [-2, 2]]

    sampler = sampler_cls(
        types=types,
        params=params,
        compute_QoIs=sigmoid_qoi,
        n_out=2
    )

    sampler.sampling(N=20, DOE_type='QRS')
    print(f"Samples shape: {sampler.X.shape}")
    print(f"QoI shape: {sampler.Y.shape}")
    assert sampler.X.shape == (20, 2)
    assert sampler.Y.shape == (20, 2)
    # Sigmoid outputs must be in (0,1)
    assert np.all((sampler.Y[:, 1] > 0) & (sampler.Y[:, 1] < 1)), "Sigmoid QoI out of bounds"
    print("✓ Sigmoid QoI test passed")


def test_sampler_normalisation():
    """Test normalization and denormalization"""
    print("\n=== Test Normalization/Denormalization ===")
    types = ['uniform', 'log_uniform']
    params = [[0, 10], [0, 5]]
    
    sampler = sampler_cls(
        types=types,
        params=params,
        n_out=1
    )
    
    # Create test samples in physical space
    X_phys = np.array([
        [5, np.exp(2.5)],
        [2.5, np.exp(1)]
    ])
    
    # Normalize
    X_norm = sampler.normalise(X_phys)
    print(f"Physical samples:\n{X_phys}")
    print(f"Normalized samples:\n{X_norm}")
    
    # Denormalize
    X_phys_recovered = sampler.denormalise(X_norm)
    print(f"Recovered physical samples:\n{X_phys_recovered}")
    
    # Check roundtrip accuracy
    error = np.abs(X_phys - X_phys_recovered)
    print(f"Max error: {np.max(error):.2e}")
    assert np.allclose(X_phys, X_phys_recovered, atol=1e-10), "Normalization roundtrip failed"
    print("✓ Normalization test passed")


def test_sampler_incremental_sampling():
    """Test incremental sampling with as_additional_points"""
    print("\n=== Test Incremental Sampling ===")
    types = ['uniform']
    params = [[0, 1]]
    
    sampler = sampler_cls(
        types=types,
        params=params,
        compute_QoIs=simple_qoi_1d,
        n_out=2
    )
    
    # First batch
    sampler.sampling(N=5, DOE_type='PRS', as_additional_points=False)
    print(f"After first sampling: {sampler.X.shape}")
    assert sampler.X.shape == (5, 1), "First batch shape mismatch"
    
    # Second batch as additional
    sampler.sampling(N=5, DOE_type='PRS', as_additional_points=True)
    print(f"After second sampling: {sampler.X.shape}")
    assert sampler.X.shape == (10, 1), "Incremental sampling failed"
    print("✓ Incremental sampling test passed")


def test_doe_additional_points():
    """Test DOE with as_additional_points"""
    print("\n=== Test DOE: Additional Points ===")
    doe = DOE_cls(dim=2, DOE_type="PRS")
    X1 = doe.sample(N=5, as_additional_points=False)
    print(f"First batch: {X1.shape}")
    
    X2 = doe.sample(N=3, as_additional_points=True)
    print(f"Second batch returned: {X2.shape}")
    print(f"Stored in DOE: {doe.X.shape}")
    
    assert doe.X.shape == (8, 2), "Additional points not stored correctly"
    print("✓ DOE additional points test passed")


def test_sampler_sg():
    """Test sampler with sparse grid DOE"""
    print("\n=== Test Sampler: Sparse Grid ===")
    types = ['uniform', 'uniform']
    params = [[-2, 2], [-2, 2]]
    
    sampler = sampler_cls(
        types=types,
        params=params,
        compute_QoIs=simple_qoi,
        n_out=2
    )
    
    sampler.sampling(N=3, DOE_type='SG')
    print(f"Samples shape: {sampler.X.shape}")
    print(f"QoI shape: {sampler.Y.shape}")
    assert sampler.X.shape[1] == 2, "Dimension mismatch"
    assert sampler.Y.shape[1] == 2, "QoI dimension mismatch"
    
    # Plot the sparse grid samples
    print("Plotting sparse grid samples...")
    prediction_plot(X=sampler.X, y=sampler.Y[:, 0], xlabel='Param 1', ylabel='Param 2', clabel='Sum', show=False)
    
    print("✓ Sparse grid sampler test passed")


def main():
    print("="*60)
    print("SAMPLER AND DOE TESTS")
    print("="*60)
    
    # DOE tests
    test_doe_prs()
    test_doe_lhs()
    test_doe_qrs()
    test_doe_sg()
    
    # Sampler tests
    test_sampler_uniform()
    test_sampler_lognormal()
    test_sampler_normalisation()
    test_sampler_incremental_sampling()
    test_doe_additional_points()
    test_sampler_sg()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    
    # Generate comprehensive 2D plots
    plot_all_tests()


def plot_all_tests():
    """Generate comprehensive 2D plots showing all sampling methods"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE 2D PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Sampler and DOE Methods Comparison', fontsize=16, fontweight='bold')
    
    # Test 1: PRS
    print("\nPlotting PRS samples...")
    doe_prs = DOE_cls(dim=2, DOE_type="PRS")
    X_prs = doe_prs.sample(N=50)
    axes[0, 0].scatter(X_prs[:, 0], X_prs[:, 1], alpha=0.6, s=50)
    axes[0, 0].set_title('PRS (Pseudo-Random)', fontweight='bold')
    axes[0, 0].set_xlabel('Param 1')
    axes[0, 0].set_ylabel('Param 2')
    axes[0, 0].grid()
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    
    # Test 2: LHS
    print("Plotting LHS samples...")
    doe_lhs = DOE_cls(dim=2, DOE_type="LHS")
    X_lhs = doe_lhs.sample(N=50)
    axes[0, 1].scatter(X_lhs[:, 0], X_lhs[:, 1], alpha=0.6, s=50, color='orange')
    axes[0, 1].set_title('LHS (Latin Hypercube)', fontweight='bold')
    axes[0, 1].set_xlabel('Param 1')
    axes[0, 1].set_ylabel('Param 2')
    axes[0, 1].grid()
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Test 3: QRS
    print("Plotting QRS samples...")
    doe_qrs = DOE_cls(dim=2, DOE_type="QRS")
    X_qrs = doe_qrs.sample(N=50)
    axes[0, 2].scatter(X_qrs[:, 0], X_qrs[:, 1], alpha=0.6, s=50, color='green')
    axes[0, 2].set_title('QRS (Sobol Quasi-Random)', fontweight='bold')
    axes[0, 2].set_xlabel('Param 1')
    axes[0, 2].set_ylabel('Param 2')
    axes[0, 2].grid()
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    
    # Test 4: Sparse Grid
    print("Plotting Sparse Grid samples...")
    doe_sg = DOE_cls(dim=2, DOE_type="SG")
    X_sg = doe_sg.sample(N=3)
    axes[1, 0].scatter(X_sg[:, 0], X_sg[:, 1], alpha=0.6, s=80, color='red', marker='s')
    axes[1, 0].set_title('Sparse Grid (Level 3)', fontweight='bold')
    axes[1, 0].set_xlabel('Param 1')
    axes[1, 0].set_ylabel('Param 2')
    axes[1, 0].grid()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # Test 5: Uniform sampler with QoI coloring
    print("Plotting Uniform distribution with QoI response...")
    sampler_uniform = sampler_cls(
        types=['uniform', 'uniform'],
        params=[[0, 1], [0, 2]],
        compute_QoIs=simple_qoi,
        n_out=2
    )
    sampler_uniform.sampling(N=30, DOE_type='LHS')
    scatter = axes[1, 1].scatter(sampler_uniform.X[:, 0], sampler_uniform.X[:, 1], 
                                  c=sampler_uniform.Y[:, 0], cmap='viridis', s=80, alpha=0.7)
    axes[1, 1].set_title('Uniform Distribution (colored by Sum QoI)', fontweight='bold')
    axes[1, 1].set_xlabel('Param 1')
    axes[1, 1].set_ylabel('Param 2')
    axes[1, 1].grid()
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Sum QoI')
    
    # Test 6: Comparison of all methods in one plot
    print("Plotting method comparison overlay...")
    axes[1, 2].scatter(X_prs[:, 0], X_prs[:, 1], alpha=0.4, s=40, label='PRS', color='blue')
    axes[1, 2].scatter(X_lhs[:, 0], X_lhs[:, 1], alpha=0.4, s=40, label='LHS', color='orange')
    axes[1, 2].scatter(X_qrs[:, 0], X_qrs[:, 1], alpha=0.4, s=40, label='QRS', color='green')
    axes[1, 2].scatter(X_sg[:, 0], X_sg[:, 1], alpha=0.8, s=100, label='SG', 
                       color='red', marker='s', edgecolors='darkred', linewidth=2)
    axes[1, 2].set_title('Method Comparison (Overlay)', fontweight='bold')
    axes[1, 2].set_xlabel('Param 1')
    axes[1, 2].set_ylabel('Param 2')
    axes[1, 2].grid()
    axes[1, 2].legend()
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sampler_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'sampler_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()
