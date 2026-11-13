"""Derivatives of the gaussian -> physical map and out-of-bounds defaulting."""
import warnings

import numpy as np
import pytest

from surroptim.param_processor import params_cls


def make_params():
    init_params = {
        "E": 210.0,
        "nu": 0.30,
        "A": np.array([0.2, 5.0, 7.0]),
    }
    active_specs = {
        "E": {"lower": 100.0, "upper": 300.0, "scale": "linear"},
        "A0": {"param": "A", "select": np.array([1, 0, 0], bool), "lower": 0.0, "upper": 1.0, "scale": "linear"},
        "A1": {"param": "A", "select": np.array([0, 1, 1], bool), "lower": 1e-3, "upper": 1e2, "scale": "log"},
    }
    return params_cls(init_params=init_params, active_specs=active_specs)


def test_active_names_follow_packed_order():
    P = make_params()
    assert P.active_names == ["E", "A0", "A1"]
    assert P.dim == 4  # A1 selects two entries


@pytest.mark.parametrize("g", [np.zeros(4), np.array([0.7, -1.3, 2.1, -0.4])])
def test_jacobian_matches_central_differences(g):
    P = make_params()
    h = 1e-6
    fd = np.array([
        (P.gauss_to_phys(g + h * e) - P.gauss_to_phys(g - h * e))[k] / (2 * h)
        for k, e in enumerate(np.eye(P.dim))
    ])
    np.testing.assert_allclose(P.gauss_to_phys_jacobian(g), fd, rtol=1e-7)


def test_jacobian_batch_shape():
    P = make_params()
    G = np.random.default_rng(0).normal(size=(5, P.dim))
    D = P.gauss_to_phys_jacobian(G)
    assert D.shape == G.shape
    np.testing.assert_allclose(D[2], P.gauss_to_phys_jacobian(G[2]))


def test_phys_grad_to_gauss_grad_chain_rule():
    P = make_params()
    g = np.array([0.3, -0.5, 1.0, 0.2])
    weights = np.array([1.0, -2.0, 0.5, 3.0])

    def f_phys(x):
        return float(weights @ x)

    def f_gauss(gg):
        return f_phys(P.gauss_to_phys(gg))

    grad = P.phys_grad_to_gauss_grad(g, weights)
    h = 1e-6
    fd = np.array([(f_gauss(g + h * e) - f_gauss(g - h * e)) / (2 * h) for e in np.eye(P.dim)])
    np.testing.assert_allclose(grad, fd, rtol=1e-7)

    grad_dict = P.phys_grad_to_gauss_grad(g, {"E": 1.0, "A0": -2.0, "A1": [0.5, 3.0]})
    np.testing.assert_allclose(grad_dict, grad)


def test_phys_grad_dict_requires_every_active_name():
    P = make_params()
    with pytest.raises(KeyError):
        P.phys_grad_to_gauss_grad(np.zeros(P.dim), {"E": 1.0})


def test_out_of_bounds_defaults_to_midpoint_with_warning():
    P = make_params()
    x = np.array([350.0, 0.5, 1e3, 1.0])   # E above, A1[0] above its bounds
    with pytest.warns(UserWarning, match="outside its bounds") as record:
        g = P.physical_to_gauss(x)
    assert len(record) == 2
    phys = P.gauss_to_phys(g)
    np.testing.assert_allclose(phys[0], 200.0)                 # linear midpoint
    np.testing.assert_allclose(phys[2], np.sqrt(1e-3 * 1e2))   # log midpoint
    np.testing.assert_allclose(phys[[1, 3]], x[[1, 3]])        # in-bounds values kept


def test_in_bounds_values_do_not_warn():
    P = make_params()
    x = np.array([150.0, 0.5, 1.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = P.physical_to_gauss(x)
    np.testing.assert_allclose(P.gauss_to_phys(g), x)


def test_clip_keeps_clipping_behaviour():
    P = make_params()
    x = np.array([350.0, 0.5, 1.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = P.physical_to_gauss(x, clip=True)
    assert P.gauss_to_phys(g)[0] == pytest.approx(300.0, rel=1e-6)


def test_value_on_bound_warns_and_is_kept():
    P = make_params()
    x = np.array([300.0, 0.0, 1.0, 10.0])   # E on its upper bound, A0 on its lower bound
    with pytest.warns(UserWarning, match="exactly on a bound") as record:
        g = P.physical_to_gauss(x)
    assert len(record) == 2
    np.testing.assert_allclose(P.gauss_to_phys(g), x, rtol=1e-6, atol=1e-9)
