[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kerfriden/SurrOptim/blob/main/demo_surroptim.ipynb)


# SurrOptim

Simple Surrogate Modelling and Optimization Library.

## QoI (Quantity of Interest) function conventions (important)

SurrOptim calls your QoI function (`qoi_fn` / legacy `compute_QoIs`) in either **sequential** (one sample at a time) or **batch** mode (many samples at once). The expected *input shape* depends on which sampler you use.

### 1) Legacy sampler (distributions/bounds) — `sampler_old_cls`

- **Sequential mode**: QoI receives a 2D array shaped `(1, d)`.
- **Batch mode** (`sample_in_batch=True`): QoI receives a 2D array shaped `(M, d)`.

So for the legacy sampler, your QoI can safely assume it always gets a 2D array.

### 2) Params-based sampler — `sampler_new_cls` / `sampler_cls(params=...)`

This sampler delegates parameter packing/unpacking to the provided `params_cls` processor.

- **Batch mode** (`batch_computation=True`, or legacy alias `sample_in_batch=True`): QoI receives a 2D array shaped `(M, d)` (for array-mode params). Output is normalized internally so that `sampler.Y` becomes `(M, n_out)`.
- **Sequential mode**: QoI may receive a 1D array shaped `(d,)` by default.

Many ML models (scikit-learn, PyTorch wrappers, etc.) expect 2D inputs even for a single sample. For these cases, construct the sampler with `qoi_force_2d=True` to ensure QoI always receives `(1, d)` for single-sample calls (including the initial call used to auto-detect `n_out`).

### Recommended robust pattern (works everywhere)

Write QoIs that accept both 1D and 2D inputs by normalizing at the top:

```python
def my_qoi(X):
	X = np.asarray(X, dtype=float)
	if X.ndim == 1:
		X = X.reshape(1, -1)  # always (M, d)

	# ... compute one QoI row per input row ...
	y = X.sum(axis=1)
	return y  # (M,) or (M,1) are both acceptable
```

### QoI outputs

- If your QoI returns an array-like, SurrOptim stores results in `sampler.Y` as a 2D array shaped `(M, n_out)`.
- In batch mode, common outputs such as `(M,)`, `(M,1)`, `(1,M)` (scalar QoI), and `(M,k)` are accepted and normalized.
- If your QoI returns a `dict`, the params-based sampler flattens it into columns in a deterministic key layout and exposes helper methods like `sampler.qoi_slices(key)`.
## Running Tests

SurrOptim has a comprehensive test suite that can be run with different optional dependencies.

### Basic Tests (No Optional Dependencies)

To run the core tests without scikit-learn or PyTorch:

```bash
python -m pytest tests/ -v
```

Expected result: **37 passed, 9 skipped**
- The 9 skipped tests require scikit-learn (6 tests) or PyTorch (3 tests)

### With Scikit-Learn

Scikit-learn is required for Gaussian Process and k-Nearest Neighbors metamodels.

```bash
pip install scikit-learn
python -m pytest tests/ -v
```

Expected result: **37 passed, 9 skipped**
- Note: Tests in `test_metamodels_all.py` are currently marked as skipped due to conftest logic
- The 9 skipped tests are primarily PyTorch-dependent neural network tests

### With PyTorch

PyTorch is required for neural network metamodels with gradient computation.

**Windows Users**: See [DOCS/SURROPTIM_CHANGELOG_AND_PYTORCH_WINDOWS.md](DOCS/SURROPTIM_CHANGELOG_AND_PYTORCH_WINDOWS.md) for PyTorch installation instructions and DLL troubleshooting.

```bash
pip install torch
python -m pytest tests/ -v
```

Expected result (if PyTorch loads successfully): **40 passed, 6 skipped**
- The 6 skipped tests would be sklearn-only tests

Expected result (if PyTorch has DLL issues on Windows): **37 passed, 9 skipped**

### With Both Dependencies

For the complete test suite:

```bash
pip install scikit-learn torch
python -m pytest tests/ -v
```

Expected result: **40+ passed, 0-6 skipped** (depending on PyTorch DLL availability on Windows)

### Test Categories

- **Core functionality**: DOE strategies, parameter processors, coordinate transformations, samplers
- **Sparse grid hierarchical sampling**: Incremental refinement from level N to N+1
- **Metamodels**: Polynomial (ridge/lasso), Gaussian Process (sklearn), k-NN (sklearn), Neural Networks (PyTorch)
- **Gradient computation**: Neural network gradients validated against finite differences (PyTorch required)