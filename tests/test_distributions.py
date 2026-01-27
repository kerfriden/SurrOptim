"""Tests for distribution strategies."""
import numpy as np

from surroptim.distributions import UniformDistribution, LogUniformDistribution


class TestUniformDistribution:
    """Test uniform distribution normalization and denormalization."""

    def test_denormalise(self):
        """Test denormalization from [-1,1] to [a,b]."""
        dist = UniformDistribution()
        params = [0.0, 10.0]  # a=0, b=10
        
        # Test center
        X_norm = np.array([0.0])
        X_physical = dist.denormalise(X_norm, params)
        assert np.isclose(X_physical[0], 5.0)
        
        # Test boundaries
        X_norm = np.array([-1.0, 1.0])
        X_physical = dist.denormalise(X_norm, params)
        assert np.isclose(X_physical[0], 0.0)
        assert np.isclose(X_physical[1], 10.0)

    def test_normalise(self):
        """Test normalization from [a,b] to [-1,1]."""
        dist = UniformDistribution()
        params = [0.0, 10.0]  # a=0, b=10
        
        # Test boundaries
        X_physical = np.array([0.0, 10.0])
        X_norm = dist.normalise(X_physical, params)
        assert np.isclose(X_norm[0], -1.0)
        assert np.isclose(X_norm[1], 1.0)
        
        # Test center
        X_physical = np.array([5.0])
        X_norm = dist.normalise(X_physical, params)
        assert np.isclose(X_norm[0], 0.0)

    def test_roundtrip(self):
        """Test roundtrip for uniform distribution."""
        dist = UniformDistribution()
        params = [1.0, 5.0]  # a=1, b=5
        
        X_physical_original = np.array([1.5, 3.0, 4.5])
        X_norm = dist.normalise(X_physical_original, params)
        X_physical_recovered = dist.denormalise(X_norm, params)
        
        np.testing.assert_allclose(X_physical_recovered, X_physical_original, rtol=1e-10)


class TestLogUniformDistribution:
    """Test log-uniform distribution normalization and denormalization."""

    def test_denormalise(self):
        """Test denormalization from [-1,1] to physical space [a,b]."""
        dist = LogUniformDistribution()
        params = [1.0, 100.0]  # physical bounds
        
        X_norm = np.array([-1.0, 0.0, 1.0])
        X_physical = dist.denormalise(X_norm, params)
        
        # Should map to approximately [1, 100]
        assert np.all(X_physical >= 1.0 - 1e-6)
        assert np.all(X_physical <= 100.0 + 1e-6)

    def test_normalise(self):
        """Test normalization from physical space to [-1,1]."""
        dist = LogUniformDistribution()
        params = [1.0, 100.0]  # physical bounds
        
        X_physical = np.array([1.0, 10.0, 100.0])
        X_norm = dist.normalise(X_physical, params)

        assert np.allclose(X_norm, np.array([-1.0, 0.0, 1.0]), atol=1e-10)

    def test_roundtrip(self):
        """Test roundtrip for log-uniform distribution."""
        dist = LogUniformDistribution()
        params = [0.1, 10.0]  # physical bounds
        
        X_physical_original = np.array([0.5, 1.0, 5.0])
        X_norm = dist.normalise(X_physical_original, params)
        X_physical_recovered = dist.denormalise(X_norm, params)
        
        np.testing.assert_allclose(X_physical_recovered, X_physical_original, rtol=1e-5)


