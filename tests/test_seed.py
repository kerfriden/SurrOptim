"""Test seed functionality in sampler."""
import numpy as np
from surroptim.sampler import sampler_legacy_cls


def simple_qoi(X):
    """Simple QoI for testing."""
    return np.sum(X, axis=1, keepdims=True)


def test_same_seed_gives_reproducible_results():
    """Test that same seed produces identical samples."""
    s1 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, seed=42, n_out=1)
    s1.sample(N=5, plot=False)
    X1 = s1.X.copy()

    s2 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, seed=42, n_out=1)
    s2.sample(N=5, plot=False)
    X2 = s2.X.copy()

    assert np.allclose(X1, X2), "Same seed should produce identical results"


def test_different_seed_gives_different_results():
    """Test that different seed produces different samples."""
    s1 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, seed=42, n_out=1)
    s1.sample(N=5, plot=False)
    X1 = s1.X.copy()

    s3 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, seed=123, n_out=1)
    s3.sample(N=5, plot=False)
    X3 = s3.X.copy()

    assert not np.allclose(X1, X3), "Different seed should produce different results"


def test_no_seed_gives_random_behavior():
    """Test that no seed produces non-reproducible results."""
    s4 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, n_out=1)
    s4.sample(N=5, plot=False)
    X4 = s4.X.copy()

    s5 = sampler_legacy_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=simple_qoi, n_out=1)
    s5.sample(N=5, plot=False)
    X5 = s5.X.copy()

    # Note: This could theoretically fail due to random chance, but probability is negligible
    assert not np.allclose(X4, X5), "No seed should produce non-reproducible results"
