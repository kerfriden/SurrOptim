import numpy as np
from surroptim.param_processor import params_cls

init_params = np.array([1.0, 0.0, 0.0, 0.0])
active_specs = {
    "a0": {"select": np.array([1, 0, 0, 0], bool), "lower": np.exp(-2.0), "upper": np.exp(2.0), "scale": "log"},
    "a1": {"select": np.array([0, 1, 1, 0], bool), "lower": [-2.0, -2.0], "upper": [2.0, 2.0], "scale": "linear"},
}

p = params_cls(init_params=init_params, active_specs=active_specs)
print('mode:', p._mode)
print('dim:', p.dim)
print('\n_layout:')
for it in p._layout:
    print(it)

print('\nparameter_slices("a0") ->', p.parameter_slices('a0'))
print('parameter_slices(["a0","a1"]) ->', p.parameter_slices(['a0','a1']))
print('\nparameter_indices("a0") ->', p.parameter_indices('a0'))
print('parameter_indices(["a0","a1"]) ->', p.parameter_indices(['a0','a1']))

# Also show pack/unpack behavior
x = p.pack(p.base)
print('\npacked base:', x)
print('unpack(packed) ->', p.unpack(x))
