Summary of recent changes

- sampler: fixed indentation corruption; added `sampler_new_cls` features:
  - `qoi_receive_packed` to control whether QoI receives packed arrays or expanded full arrays.
  - `_expand_packed_to_full` helper to expand packed array-mode vectors to full base arrays.
  - `add_to_dataset` default semantics with backward-compatible alias handling for legacy names.
  - QoI dict flattening helpers: `qoi_slices`, `qoi_indices` and layout building.
- param_processor: added plural helpers (`parameter_slices`, `parameter_indices`), convenience aliases and robust pack/unpack behavior for array-mode.
- neural network metamodel: added `predict_and_grad` (numpy-facing regressor uses PyTorch autograd) and enforced double precision for torch conversions.

Test status (local):
- Ran non-PyTorch suite: `pytest -q --ignore=tests/test_metamodels_all.py` → `30 passed, 3 warnings` (1.90s).
- PyTorch tests require a working PyTorch installation on Windows; previously import failed with "DLL load failed while importing _C" in this environment.

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
