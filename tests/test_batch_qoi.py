import numpy as np
import warnings
from surroptim.param_processor import params_cls
from surroptim.sampler import sampler_new_cls


def _base_specs():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }
    return init_array, active_specs


def test_qoi_receives_batched_array():
    init_array, active_specs = _base_specs()
    P = params_cls(init_params=init_array, active_specs=active_specs)

    called = {"batched": False}

    def qoi_vec(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            called["batched"] = True
            # return one scalar per row
            return np.atleast_2d(a.sum(axis=1)).T
        # single sample
        return np.array([[a.sum()]])

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=0, qoi_fn=qoi_vec)
    sampler.sample(N=6, as_additional_points=False, batch_computation=True)

    assert called["batched"] is True
    assert sampler.Y.shape[0] == 6


def test_batch_fallback_warns_and_still_computes():
    init_array, active_specs = _base_specs()
    P = params_cls(init_params=init_array, active_specs=active_specs)

    # QoI that fails on 2D input by raising TypeError
    def qoi_fail_on_batch(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            raise TypeError("I don't accept batched arrays")
        return np.array([[a.sum()]])

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=1, qoi_fn=qoi_fail_on_batch)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sampler.sample(N=4, as_additional_points=False, batch_computation=True)
        # ensure a warning was emitted
        assert any(isinstance(x.message, Warning) for x in w)

    assert sampler.Y.shape[0] == 4


def test_qoi_dict_output_and_key_index():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }
    P = params_cls(init_params=init_array, active_specs=active_specs)

    def qoi_dict(arr):
        a = np.asarray(arr)
        # support batched input
        if a.ndim == 2:
            s = np.sum(a, axis=1)  # (M,)
            v = a[:, :2]  # (M,2)
            return {"s": s, "v": v}
        # single-sample
        return {"s": np.array([a.sum()]), "v": np.array([a[0], a[1]])}

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=2, qoi_fn=qoi_dict)
    sampler.sample(N=3, as_additional_points=False, batch_computation=True)

    # Check Y shape and key indices
    sl_s = sampler.qoi_slices("s")
    sl_v = sampler.qoi_slices("v")
    assert sampler.Y.shape == (3, sampler.qoi_dim)
    # Values in Y match dict entries
    assert np.allclose(sampler.Y[:, sl_s], np.atleast_2d(np.sum(sampler.X, axis=1)).T)
    assert np.allclose(sampler.Y[:, sl_v], np.vstack([sampler.X[:, 0], sampler.X[:, 1]]).T)


def test_qoi_dict_with_flat_key():
    init_array = np.array([1.0, 0.0, 0.0])
    active_specs = {
        "a0": {"select": np.array([1, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
        "a1": {"select": np.array([0, 1, 0], bool), "lower": -2.0, "upper": 2.0, "scale": "linear"},
    }
    P = params_cls(init_params=init_array, active_specs=active_specs)

    def qoi_flat(arr):
        a = np.asarray(arr)
        # batched: return flat key 'f' as (M,) array
        if a.ndim == 2:
            s = np.sum(a, axis=1)
            f = a[:, :2]  # two components per-sample
            return {"s": s, "f": f}
        return {"s": np.array([a.sum()]), "f": np.array(a[0])}

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=3, qoi_fn=qoi_flat)
    sampler.sample(N=4, as_additional_points=False, batch_computation=True)

    sl_s = sampler.qoi_slices("s")
    sl_f = sampler.qoi_slices("f")
    assert sampler.Y.shape == (4, sampler.qoi_dim)
    # multi-component key should match first two columns of X
    assert np.allclose(sampler.Y[:, sl_f], np.vstack([sampler.X[:, 0], sampler.X[:, 1]]).T)

    # multiple-keys helper
    mapping = sampler.qoi_slices(["s", "f"])
    assert mapping["s"] == sl_s
    assert mapping["f"] == sl_f

    # concatenated integer indices for keys
    idxs = sampler.qoi_indices(["s", "f"])
    expected = np.concatenate([np.arange(mapping[k].start, mapping[k].stop) for k in ["s", "f"]])
    assert np.array_equal(idxs, expected)


def test_batch_qoi_accepts_M_vector_and_normalizes_to_Mx1():
    init_array, active_specs = _base_specs()
    P = params_cls(init_params=init_array, active_specs=active_specs)

    def qoi_m_vector(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            # (M,) on purpose
            return a.sum(axis=1)
        return np.array([a.sum()])

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=10, qoi_fn=qoi_m_vector)
    sampler.sample(N=5, as_additional_points=False, batch_computation=True)

    assert sampler.Y.shape == (5, 1)
    assert np.allclose(sampler.Y[:, 0], np.sum(sampler.X, axis=1))


def test_batch_qoi_accepts_row_vector_1xM_and_normalizes_to_Mx1():
    init_array, active_specs = _base_specs()
    P = params_cls(init_params=init_array, active_specs=active_specs)

    def qoi_row_vector(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            # (1,M) on purpose
            return np.atleast_2d(a.sum(axis=1))
        return np.array([[a.sum()]])

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=11, qoi_fn=qoi_row_vector)
    sampler.sample(N=6, as_additional_points=False, batch_computation=True)

    assert sampler.Y.shape == (6, 1)
    assert np.allclose(sampler.Y[:, 0], np.sum(sampler.X, axis=1))


def test_batch_qoi_multioutput_Mx2_is_supported_and_preserved():
    init_array, active_specs = _base_specs()
    P = params_cls(init_params=init_array, active_specs=active_specs)

    def qoi_two_outputs(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            # (M,2)
            return a[:, :2]
        # (2,) for single-sample -> auto-detect n_out=2
        return np.array([a[0], a[1]])

    sampler = sampler_new_cls(params=P, DOE_type="QRS", seed=12, qoi_fn=qoi_two_outputs)
    sampler.sample(N=4, as_additional_points=False, batch_computation=True)

    assert sampler.Y.shape == (4, 2)
    assert np.allclose(sampler.Y, sampler.X[:, :2])
