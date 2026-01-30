# SurrOptim Changelog and PyTorch Windows Installation Guide

## Recent Code Improvements (January 2026)

### Major Refactoring
- **PyTorch Gradient Computation**: Vectorized backward pass in `neural_network_meta_model.py` - replaces inefficient loop over outputs with vectorized gradient computation. Performance improvement: up to N× speedup where N is batch size.
- **Coordinate Transformations**: Standardized naming convention with canonical methods:
  - `phys_to_unit`, `unit_to_phys`, `unit_to_gauss`, `gauss_to_unit`, `phys_to_gauss`, `gauss_to_phys`
  - All other variations (reference_to_physical, etc.) are now aliases
- **Code Deduplication**: Extracted helper methods for parameter resolution, sparse grid detection, and distribution transformations
- **Factory Function Fix**: Renamed factory to avoid shadowing class definitions
- **Gradient Validation Tests**: Added `test_neural_network_gradients.py` with finite difference validation

### Test Status
- **32 passed, 9 skipped** (PyTorch tests skip when unavailable)
- All core functionality works without PyTorch
- Neural network metamodels require working PyTorch installation

---

## PyTorch Installation on Windows - IMPORTANT

### The DLL Problem
PyTorch often fails on Windows with: `ImportError: DLL load failed while importing _C`

This happens because PyTorch requires **Microsoft Visual C++ Redistributable** libraries that may not be installed on your system.

### Solution Steps (try in order)

#### Option 1: Install Visual C++ Redistributable (RECOMMENDED FIRST STEP)
```powershell
# Download and install from Microsoft:
# https://aka.ms/vs/17/release/vc_redist.x64.exe
# Then restart your terminal
```

#### Option 2: Install PyTorch CPU-only (simplest, works on all PCs)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option 3: Use Conda (most reliable for Windows)
```powershell
# Create clean environment
conda create -n surroptim python=3.11 -y
conda activate surroptim

# Install PyTorch CPU-only
conda install -c pytorch pytorch cpuonly -y

# Install project
pip install -e .
```

#### Option 4: Install with CUDA (if you have NVIDIA GPU)
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation Works
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} works!')"
```

If this prints a version number, PyTorch is working correctly.

### Running Tests
```powershell
# Run all tests (PyTorch tests will skip if not available)
pytest tests/ -v

# Run only neural network gradient tests (requires PyTorch)
pytest tests/test_neural_network_gradients.py -v

# Run all tests except PyTorch-dependent ones
pytest tests/ -v --ignore=tests/test_neural_network_gradients.py --ignore=tests/test_metamodels_all.py
```

### Important Notes
- **SurrOptim works WITHOUT PyTorch** - you just can't use neural network metamodels
- Tests automatically skip PyTorch-dependent tests if import fails
- Conda is more reliable than pip for PyTorch on Windows due to bundled C++ runtimes
- CPU-only PyTorch is sufficient for most metamodeling use cases

---

## Previous Changelog

Summary of recent changes

- sampler: fixed indentation corruption; added `sampler_new_cls` features:
  - `qoi_receive_packed` to control whether QoI receives packed arrays or expanded full arrays.
  - `_expand_packed_to_full` helper to expand packed array-mode vectors to full base arrays.
  - `add_to_dataset` default semantics with backward-compatible alias handling for legacy names.
  - QoI dict flattening helpers: `qoi_slices`, `qoi_indices` and layout building.
- param_processor: added plural helpers (`parameter_slices`, `parameter_indices`), convenience aliases and robust pack/unpack behavior for array-mode.
- neural network metamodel: added `predict_and_grad` (numpy-facing regressor uses PyTorch autograd) and enforced double precision for torch conversions.

## Previous Changelog

Summary of recent changes

- sampler: fixed indentation corruption; added `sampler_new_cls` features:
  - `qoi_receive_packed` to control whether QoI receives packed arrays or expanded full arrays.
  - `_expand_packed_to_full` helper to expand packed array-mode vectors to full base arrays.
  - `add_to_dataset` default semantics with backward-compatible alias handling for legacy names.
  - QoI dict flattening helpers: `qoi_slices`, `qoi_indices` and layout building.
- param_processor: added plural helpers (`parameter_slices`, `parameter_indices`), convenience aliases and robust pack/unpack behavior for array-mode.
- neural network metamodel: added `predict_and_grad` (numpy-facing regressor uses PyTorch autograd) and enforced double precision for torch conversions.

---

## Historical PyTorch Installation Notes

Recommended PyTorch install for Windows (Conda, CPU-only):

1. Create and activate a clean conda environment (example uses Python 3.11):

```powershell
conda create -n surroptim python=3.11 -y
conda activate surroptim
```

2. Install PyTorch (CPU-only):

```powershell
conda install -c pytorch pytorch cpuonly -y
```

3. (Optional) If you have an NVIDIA GPU and want CUDA support, install the matching CUDA-enabled build. Example for CUDA 12.1:

```powershell
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
```

4. Install project dependencies and the package itself in editable mode:

```powershell
pip install -e .
# or, if you use extras in pyproject, adapt accordingly
```

5. Run the full test suite:

```powershell
pytest -q
```

Notes:
- Conda is recommended on Windows to avoid DLL/ABI issues with PyTorch wheels.
- Use the CUDA version that matches your GPU drivers if installing GPU-enabled PyTorch.

If you want, I can:
- Add these notes to `README.md` instead of `DOCS/`.
- Attempt to run the PyTorch tests here after you enable/confirm a working PyTorch runtime.
- Open a PR with the changes.
