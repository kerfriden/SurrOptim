import numpy as np
from surroptim.param_processor import params_cls


def test_parameter_slices_and_indices_dict_mode():
    init_params = {
        "E": 210.0,
        "A": np.array([0.2, 5.0, 7.0])
    }

    active_specs = {
        "E": {"lower": 100.0, "upper": 300.0, "scale": "linear", "select": None},
        "A0": {"param": "A", "select": np.array([1, 0, 0], bool), "lower": 0.0, "upper": 1.0, "scale": "linear"},
        "A1": {"param": "A", "select": np.array([0, 1, 0], bool), "lower": 1e-3, "upper": 1e2, "scale": "log"},
    }

    P = params_cls(init_params=init_params, active_specs=active_specs)

    # single slices (use plural API accepting a single key)
    sl_A0 = P.parameter_slices("A0")
    sl_A1 = P.parameter_slices("A1")

    packed = P.pack(P.base)

    # Check slice contents match packed values from A
    assert packed[sl_A0].shape[0] == 1
    assert packed[sl_A1].shape[0] == 1
    assert np.isclose(packed[sl_A0][0], init_params["A"][0])
    assert np.isclose(packed[sl_A1][0], init_params["A"][1])

    # multiple slices mapping
    mapping = P.parameter_slices(["A0", "A1"])
    assert mapping["A0"] == sl_A0 and mapping["A1"] == sl_A1

    # concatenated indices
    idxs = P.parameter_indices(["A0", "A1"])
    expected = np.concatenate([np.arange(sl_A0.start, sl_A0.stop), np.arange(sl_A1.start, sl_A1.stop)])
    assert np.array_equal(idxs, expected)
