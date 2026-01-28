import numpy as np
from surroptim.sampler import sampler_cls

def qoi(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1).reshape(-1, 1)

s = sampler_cls(['uniform', 'uniform'], [[-1, 1], [-1, 1]], qoi_fn=qoi, n_out=1)
print('calling sample(n_samples=5)')
s.sample(n_samples=5)
print('X.shape=', None if s.X is None else s.X.shape, 'Y.shape=', None if s.Y is None else s.Y.shape)
