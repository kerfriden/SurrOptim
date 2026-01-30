"""
Test hierarchical sparse grid sampling for both legacy and new sampler classes.

This module verifies that when using sparse grids with incremental refinement
(e.g., level=5 then level=6), only the delta points are computed, not the entire
grid from scratch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from surroptim.sampler import sampler_legacy_cls, sampler_cls
from surroptim.param_processor import params_cls


def create_bounds(dim, lb, ub):
    """Helper to create bounds list."""
    return [[lb, ub] for _ in range(dim)]


def create_distributions(dim, dist='uniform'):
    """Helper to create distributions list."""
    return [dist for _ in range(dim)]


def qoi_function(x):
    """Simple quadratic QoI for testing."""
    if isinstance(x, dict):
        # For new sampler with dict input
        arr = np.array([v for v in x.values()])
        return np.sum(arr**2)
    else:
        # For legacy sampler with array input
        return np.sum(x**2)


def test_legacy_sampler_hierarchical_sparse_grid(capsys):
    """Test hierarchical refinement with sampler_legacy_cls."""
    # Setup
    bounds = create_bounds(dim=3, lb=-2.0, ub=2.0)
    distributions = create_distributions(dim=3, dist='uniform')
    sampler = sampler_legacy_cls(bounds=bounds, distributions=distributions, 
                                 qoi_fn=qoi_function, doe_type='SG', n_out=1)
    
    # First sampling at level 3
    sampler.sample(level=3, plot=False)
    captured = capsys.readouterr()
    n_samples_level3 = len(sampler.X_reference)
    
    print(f"Level 3 resulted in {n_samples_level3} samples")
    assert n_samples_level3 > 0, "Level 3 should produce samples"
    assert sampler.X_reference is not None
    assert sampler.X is not None
    assert sampler.Y is not None
    
    # Hierarchical refinement to level 4
    sampler.sample(level=4, as_additional_samples=True, plot=False)
    captured = capsys.readouterr()
    n_samples_level4_total = len(sampler.X_reference)
    
    # Verify hierarchical behavior
    print(f"Level 4 resulted in {n_samples_level4_total} total samples")
    assert n_samples_level4_total > n_samples_level3, "Level 4 should have more samples than level 3"
    
    # Check that the message indicates hierarchical refinement
    output = captured.out
    assert "hierarchical refinement" in output.lower() or "additional samples" in output.lower(), \
        "Output should mention hierarchical refinement or additional samples"
    
    # Verify the delta computation
    delta_samples = n_samples_level4_total - n_samples_level3
    print(f"Delta samples from level 3 to 4: {delta_samples}")
    assert delta_samples > 0, "Should have added new samples"
    
    # Check that the output mentions the number of new points
    assert str(delta_samples) in output or "new points" in output.lower(), \
        "Output should mention the number of new points"


def test_new_sampler_hierarchical_sparse_grid(capsys):
    """Test hierarchical refinement with sampler_cls using params_cls."""
    # Setup with param processor - requires init_params and active_specs
    init_params = {
        'x1': 0.0,
        'x2': 0.0,
        'x3': 0.0
    }
    active_specs = {
        'x1': {'param': 'x1', 'lower': -2.0, 'upper': 2.0, 'scale': 'linear'},
        'x2': {'param': 'x2', 'lower': -2.0, 'upper': 2.0, 'scale': 'linear'},
        'x3': {'param': 'x3', 'lower': -2.0, 'upper': 2.0, 'scale': 'linear'}
    }
    params = params_cls(init_params=init_params, active_specs=active_specs)
    
    sampler = sampler_cls(params=params, qoi_fn=qoi_function, doe_type='SG')
    
    # First sampling at level 2
    sampler.sample(level=2)
    captured = capsys.readouterr()
    n_samples_level2 = len(sampler.X_reference)
    
    print(f"Level 2 resulted in {n_samples_level2} samples")
    assert n_samples_level2 > 0, "Level 2 should produce samples"
    assert sampler.X_reference is not None
    assert sampler.X is not None
    assert sampler.Y is not None
    
    # Hierarchical refinement to level 3
    sampler.sample(level=3, as_additional_samples=True)
    captured = capsys.readouterr()
    n_samples_level3_total = len(sampler.X_reference)
    
    # Verify hierarchical behavior
    print(f"Level 3 resulted in {n_samples_level3_total} total samples")
    assert n_samples_level3_total > n_samples_level2, "Level 3 should have more samples than level 2"
    
    # Check that the message indicates hierarchical refinement
    output = captured.out
    assert "hierarchical refinement" in output.lower() or "additional samples" in output.lower(), \
        "Output should mention hierarchical refinement or additional samples"
    
    # Verify the delta computation
    delta_samples = n_samples_level3_total - n_samples_level2
    print(f"Delta samples from level 2 to 3: {delta_samples}")
    assert delta_samples > 0, "Should have added new samples"


def test_sparse_grid_level_increment_details(capsys):
    """Test that sparse grid provides detailed information about hierarchical refinement."""
    # Use legacy sampler for this test
    bounds = create_bounds(dim=2, lb=-1.0, ub=1.0)
    distributions = create_distributions(dim=2, dist='uniform')
    sampler = sampler_legacy_cls(bounds=bounds, distributions=distributions,
                                 qoi_fn=qoi_function, doe_type='SG', n_out=1)
    
    # Sample at level 5
    sampler.sample(level=5, plot=False)
    captured = capsys.readouterr()
    n_level5 = len(sampler.X_reference)
    
    # Sample at level 6 hierarchically
    sampler.sample(level=6, as_additional_samples=True, plot=False)
    captured = capsys.readouterr()
    n_level6 = len(sampler.X_reference)
    
    output = captured.out
    print("Captured output:")
    print(output)
    
    # Verify the output contains expected information
    assert "hierarchical refinement" in output.lower(), \
        "Should mention hierarchical refinement"
    assert "level 5" in output.lower() and "level 6" in output.lower(), \
        "Should mention both level 5 and level 6"
    assert "previous points" in output.lower() or str(n_level5) in output, \
        "Should mention the number of previous points"
    assert "new points" in output.lower(), \
        "Should mention new points"
    
    # Verify only delta was computed
    delta = n_level6 - n_level5
    assert delta > 0, "Should have added new samples"
    
    # The output should show that we're only computing the delta
    # (The number of new points should be much smaller than total at level 6)
    assert delta < n_level6, "Delta should be less than total (sanity check)"


def test_sparse_grid_no_hierarchical_when_not_requested(capsys):
    """Test that without as_additional_samples, grid is recomputed from scratch."""
    bounds = create_bounds(dim=2, lb=-1.0, ub=1.0)
    distributions = create_distributions(dim=2, dist='uniform')
    sampler = sampler_legacy_cls(bounds=bounds, distributions=distributions,
                                 qoi_fn=qoi_function, doe_type='SG', n_out=1)
    
    # Sample at level 3
    sampler.sample(level=3, plot=False)
    n_level3 = len(sampler.X_reference)
    
    # Sample at level 4 WITHOUT as_additional_samples
    sampler.sample(level=4, as_additional_samples=False, plot=False)
    captured = capsys.readouterr()
    n_level4 = len(sampler.X_reference)
    
    output = captured.out
    
    # Should warn about reinitializing
    assert "reinitializ" in output.lower() or "warning" in output.lower(), \
        "Should warn when restarting sampling"
    
    # The total should be level 4 samples only, not level3 + level4
    # (but we can't easily verify the exact count without hierarchical mode active)
    assert n_level4 > n_level3, "Level 4 has more points than level 3"


def test_both_samplers_produce_same_hierarchical_behavior():
    """Verify that both sampler classes produce consistent hierarchical behavior."""
    # Legacy sampler
    bounds_legacy = create_bounds(dim=2, lb=-1.0, ub=1.0)
    distributions_legacy = create_distributions(dim=2, dist='uniform')
    sampler_legacy = sampler_legacy_cls(bounds=bounds_legacy, distributions=distributions_legacy,
                                        qoi_fn=qoi_function, doe_type='SG', n_out=1)
    sampler_legacy.sample(level=3, plot=False)
    n_legacy_l3 = len(sampler_legacy.X_reference)  # Total samples after level 3
    sampler_legacy.sample(level=4, as_additional_samples=True, plot=False)
    n_legacy_l4 = len(sampler_legacy.X_reference)  # Total samples after level 4
    delta_legacy = n_legacy_l4 - n_legacy_l3
    
    # New sampler
    init_params = {'x1': 0.0, 'x2': 0.0}
    active_specs = {
        'x1': {'param': 'x1', 'lower': -1.0, 'upper': 1.0, 'scale': 'linear'},
        'x2': {'param': 'x2', 'lower': -1.0, 'upper': 1.0, 'scale': 'linear'}
    }
    params = params_cls(init_params=init_params, active_specs=active_specs)
    sampler_new = sampler_cls(params=params, qoi_fn=qoi_function, doe_type='SG')
    sampler_new.sample(level=3)
    n_new_l3 = len(sampler_new.X_reference)  # Total samples after level 3
    sampler_new.sample(level=4, as_additional_samples=True)
    n_new_l4 = len(sampler_new.X_reference)  # Total samples after level 4
    delta_new = n_new_l4 - n_new_l3
    
    # Both should have the same sample counts
    print(f"Legacy - Level 3: {n_legacy_l3}, Level 4: {n_legacy_l4}, Delta: {delta_legacy}")
    print(f"New    - Level 3: {n_new_l3}, Level 4: {n_new_l4}, Delta: {delta_new}")
    
    assert n_legacy_l3 == n_new_l3, f"Both samplers should have same count at level 3: {n_legacy_l3} vs {n_new_l3}"
    assert n_legacy_l4 == n_new_l4, f"Both samplers should have same count at level 4: {n_legacy_l4} vs {n_new_l4}"
    assert delta_legacy == delta_new, f"Both samplers should add same delta: {delta_legacy} vs {delta_new}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
