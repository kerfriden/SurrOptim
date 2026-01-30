[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kerfriden/SurrOptim/blob/main/demo_surroptim.ipynb)


# SurrOptim

Simple Surrogate Modelling and Optimization Library.
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