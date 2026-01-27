# SurrOptim AI Coding Instructions

## Project Overview
SurrOptim is a Python optimization library focused on parameter transformation pipelines for optimization algorithms. The core is a bidirectional transformation system: `dict ↔ physical ↔ unit ↔ gaussian` spaces.

## Architecture

### Core Component: ParameterProcessor ([param_processor.py](../src/param_processor.py))
The single-file implementation handles parameter space transformations with support for:
- **Multiple scales**: linear and log (with distinct math for each)
- **Selective activation**: Use `select` masks to optimize subsets of array parameters
- **Batch operations**: All methods accept single `(N,)` or batch `(M,N)` arrays
- **Overlap resolution**: Last-wins strategy in `_trim_selects_last_wins` - later specs override earlier ones

### Key Design Patterns

**Dual input/output modes**: Every transformation method accepts and returns appropriate formats:
```python
# Single: dict -> (N,) or (N,) -> dict
x = P.pack(params_dict)          # Returns (N,) array
params = P.unpack(x)              # Returns dict

# Batch: list[dict] -> (M,N) or (M,N) -> list[dict]  
X = P.pack([dict1, dict2, ...])   # Returns (M,N) array
param_list = P.unpack(X)          # Returns list[dict]
```

**Transformation chain invariants**:
- `physical_to_unit` maps `[lower, upper]` → `[-1, 1]` (respecting scale)
- `unit_to_gauss` uses `Φ⁻¹((z+1)/2)` to map `[-1,1]` → unbounded gaussian
- Inverse path must satisfy roundtrip: `x ≈ gauss_to_physical(physical_to_gauss(x))`

**Log-scale handling**: When `scale="log"`, bounds must be `> 0`:
```python
# Linear: y = lower + t*(upper - lower), where t ∈ [0,1]
# Log: y = exp(log(lower) + t*(log(upper) - log(lower)))
```

## Critical Conventions

1. **Scalar preservation**: Keys in `_scalar_keys` are cast back to Python `float` (not numpy arrays) in `unpack()` for cleaner dict output

2. **Select semantics**:
   - `select=None` → entire parameter is active
   - Boolean array → must match parameter shape exactly
   - Integer/slice indexing → converted to boolean mask internally

3. **Spec structure**: Active specs require `lower`, `upper`; optional `param` (defaults to var_id), `scale` (defaults to "linear"), `select` (defaults to None)

4. **Normal distribution functions**:
   - `norm_cdf`: Uses `np.erf` when available, falls back to `math.erf`
   - `norm_ppf`: Prefers `np.erfinv`, otherwise Acklam approximation for accuracy

## Development Workflow

**Setup** (from [README.md](../README.md)):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -U pip
pip install -e .
```

**Testing**: Run the tests in [test_param_cls.py](../tests/test_param_cls.py):
```bash
python tests/test_param_cls.py
```
Tests verify roundtrip accuracy (`rtol=1e-7, atol=1e-9`) for single/batch modes across all transformation chains.

**Package structure**: Uses `setuptools` with `src/` layout - all code lives in `src/`

## When Modifying Code

- Preserve the layered transformation design - don't shortcut chains (e.g., add `dict_to_gauss` as `unit_to_gauss(physical_to_unit(pack(...)))`, not a new implementation)
- Maintain batch/single duality in all new methods using `_as_X()` pattern
- For new scales, add math to both `physical_to_unit` and `unit_to_physical` with corresponding pre-computed fields in `_layout` (like `loglo`/`loghi`)
- Test roundtrip accuracy when changing transformation math - numerical precision matters for optimization
- The `clip` parameter guards against bounds violations; use conservatively (typically only when dealing with external/untrusted inputs)
