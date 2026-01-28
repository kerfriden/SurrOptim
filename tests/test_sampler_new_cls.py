import numpy as np
from surroptim.param_processor import params_cls
from surroptim.sampler import sampler_new_cls

# ============================================================================
# 2. Incremental sampling with QoI evaluation using a single 3-element 'a'
# ============================================================================
print("\n=== Test 2: Incremental Sampling with Sigmoid QoI (vector 'a') ===")


def sigmoid_qoi(params_dict) -> np.ndarray:
    """QoI: [sum, sigmoid] where x=params['a'][0], y=params['a'][1]
    Uses scalar `power` from params_dict (default 2.0) as exponent for y."""
    A = params_dict.get("a")
    if A is None:
        raise KeyError("expected parameter 'a' in params dict")
    x = float(A[0])
    y = float(A[1])
    p = float(params_dict.get("power", 2.0))
    s = np.log(x) + y ** p
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


def test_sampler_new_cls_incremental_sigmoid():
    # init params: single 3-element array 'a' with two active components (indices 0 and 1)
    init_params = {"a": np.array([1.0, 0.0, 0.0]), "power": 2.0}
    active_specs = {
        "a0": {"param": "a", "select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"param": "a", "select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }

    P = params_cls(init_params=init_params, active_specs=active_specs)

    sampler = sampler_new_cls(
        params=P,
        DOE_type="QRS",
        seed=0,
        qoi_fn=sigmoid_qoi,
    )

    # first batch
    sampler.sample(N=8, as_additional_points=False)
    assert sampler.X.shape[0] == 8 and sampler.X.shape[1] == 2
    assert sampler.Y.shape == (8, 2)

    # second batch appended
    sampler.sample(N=8, as_additional_points=True)
    assert sampler.X.shape[0] == 16 and sampler.X.shape[1] == 2
    assert sampler.Y.shape == (16, 2)

    # check ranges: x positive (log-uniform), y in [-2,2], sigmoid in (0,1)
    assert np.all(sampler.X[:, 0] > 0.0)
    assert np.all(sampler.X[:, 1] >= -2.0) and np.all(sampler.X[:, 1] <= 2.0)
    assert np.all((sampler.Y[:, 1] > 0.0) & (sampler.Y[:, 1] < 1.0))
