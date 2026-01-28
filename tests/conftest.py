import pytest
from pathlib import Path

try:
    import sklearn  # noqa: F401
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """If scikit-learn is not installed, automatically mark tests that
    import or reference sklearn to be skipped. This allows running the
    test suite in minimal environments.
    """
    # Create skip markers for any missing optional dependencies so both
    # scikit-learn and PyTorch are treated symmetrically.
    sklearn_msg = "skipping test because scikit-learn is not installed"
    torch_msg = "skipping test because PyTorch is not installed"
    sklearn_marker = pytest.mark.skip(reason=sklearn_msg)
    torch_marker = pytest.mark.skip(reason=torch_msg)

    for item in items:
        try:
            src = Path(item.fspath).read_text()
        except Exception:
            continue

        if (not _SKLEARN_AVAILABLE) and ("from sklearn" in src or "import sklearn" in src):
            item.add_marker(sklearn_marker)
            # continue so torch-related checks still run in case both are missing

        if (not _TORCH_AVAILABLE) and (
            "neural_net" in src or "neural_network" in src or "import torch" in src
        ):
            item.add_marker(torch_marker)