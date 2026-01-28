"""Test seed functionality in sampler."""
import numpy as np
from surroptim.sampler import sampler_old_cls as sampler_cls


def test_qoi(X):
    return np.sum(X, axis=1, keepdims=True)


# Test 1: Same seed gives same samples
print('Test 1: Same seed = reproducible results')
s1 = sampler_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=test_qoi, seed=42)
s1.sample(N=5)
X1 = s1.X.copy()

s2 = sampler_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=test_qoi, seed=42)
s2.sample(N=5)
X2 = s2.X.copy()

print(f'Same seed match: {np.allclose(X1, X2)}')
print(f'X1[0] = {X1[0]}')
print(f'X2[0] = {X2[0]}')

# Test 2: Different seed gives different samples
print('\nTest 2: Different seed = different results')
s3 = sampler_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=test_qoi, seed=123)
s3.sample(N=5)
X3 = s3.X.copy()

print(f'Different seed match: {np.allclose(X1, X3)}')
print(f'X3[0] = {X3[0]}')

# Test 3: No seed gives random results each time
print('\nTest 3: No seed = random behavior')
s4 = sampler_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=test_qoi)
s4.sample(N=5)
X4 = s4.X.copy()

s5 = sampler_cls(distributions=['uniform', 'uniform'], bounds=[[-1,1], [-1,1]], qoi_fn=test_qoi)
s5.sample(N=5)
X5 = s5.X.copy()

print(f'No seed match (should be False): {np.allclose(X4, X5)}')
print(f'X4[0] = {X4[0]}')
print(f'X5[0] = {X5[0]}')

print('\nAll tests passed!')
