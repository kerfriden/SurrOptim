import numpy as np
from surroptim.param_processor import params_cls
from surroptim.sampler import sampler_cls


def test_array_init_pack_unpack_and_transforms():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }

    P = params_cls(init_params=init_array, active_specs=active_specs)

    # dim corresponds to two active entries
    assert P.dim == 2

    # pack/unpack via dict interface (base param named '__arr')
    packed = P.pack({"__arr": init_array})
    assert packed.shape == (2,)

    rec = P.unpack(packed)
    assert isinstance(rec, np.ndarray)
    assert rec.shape == (2,)

    # physical <-> reference roundtrip
    z = P.physical_to_reference(packed)
    x2 = P.reference_to_physical(z)
    assert np.allclose(packed, x2)


def test_array_init_sampler_compatibility():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }

    P = params_cls(init_params=init_array, active_specs=active_specs)

    # QoI expects a params-dict and reads '__arr'
    # QoI expects a numpy array input when params created from ndarray
    def qoi_arr(arr):
        s = float(arr[0]) + float(arr[1])
        return np.array([[s, 0.5]])

    sampler = sampler_cls(params=P, DOE_type="QRS", seed=0, qoi_fn=qoi_arr)
    sampler.sample(N=4, as_additional_points=False, batch_computation=True)

    assert sampler.X.shape == (4, 2)
    assert sampler.Y.shape[0] == 4


def test_array_init_variant_select_and_vector_bounds():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        # second spec activates two entries (indices 1 and 2) with per-entry bounds
        "a1": {"select": np.array([0, 1, 1], bool), "lower": [-2.0, -2.0], "upper": [2.0, 1.0], "scale": "linear"},
    }

    P = params_cls(init_params=init_array, active_specs=active_specs)

    # two specs: a0 contributes 1, a1 contributes 2 -> dim == 3
    assert P.dim == 3

    # pack via dict interface and unpack back
    packed = P.pack({"__arr": init_array})
    assert packed.shape == (3,)

    rec = P.unpack(packed)
    assert isinstance(rec, np.ndarray)
    assert rec.shape == (3,)

    # check per-entry bounds mapping: reference space length should match dim
    z = P.physical_to_reference(packed)
    assert z.shape == (3,)

    # run sampler compatibility: QoI reads '__arr'
    # QoI accepts flat numpy array for array-mode params
    def qoi_arr(arr):
        return np.array([[float(arr[0] + arr[1] + arr[2])]])

    sampler = sampler_cls(params=P, DOE_type="QRS", seed=1, qoi_fn=qoi_arr)
    sampler.sample(N=5, as_additional_points=False, batch_computation=True)

    assert sampler.X.shape == (5, P.dim)
    assert sampler.Y.shape[0] == 5
