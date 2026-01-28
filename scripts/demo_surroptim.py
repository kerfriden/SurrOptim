"""Demo script: SurrOptim sampling and metamodeling."""

import numpy as np
import matplotlib.pyplot as plt

from surroptim.sampler import sampler_old_cls as sampler_cls
from surroptim.polynomial_meta_models import polynomial_lasso_regressor

# ============================================================================
# 1. Simple sparse grid sampling
# ============================================================================
print("=== Test 1: Sparse Grid Sampling ===")
sampler_sg = sampler_cls(
    distributions=["log_uniform", "uniform"],
    bounds=[[np.exp(-2), np.exp(2.)], [-2., 2.]],
    DOE_type='SG',
    seed=0
)

sampler_sg.sample(N=2)

plt.figure(figsize=(6, 5))
plt.scatter(sampler_sg.X[:, 0], sampler_sg.X[:, 1])
plt.xlabel("x (log-uniform: [-2, 2])")
plt.ylabel("y (uniform: [-2, 2])")
plt.grid()
plt.title("Sparse Grid Samples")
plt.show()

# ============================================================================
# 1b. Sparse grid with log-uniform (requires positive bounds)
# ============================================================================
print("\n=== Test 1b: Sparse Grid with Log-Uniform ===")
sampler_sg_log = sampler_cls(
    distributions=["log_uniform", "uniform"],
    bounds=[[np.exp(-2), np.exp(2.)], [-2., 2.]],
    DOE_type='SG',
    seed=0
)

sampler_sg_log.sample(N=2)

plt.figure(figsize=(6, 5))
plt.scatter(sampler_sg_log.X[:, 0], sampler_sg_log.X[:, 1])
plt.xlabel("x (log-uniform: [-2, 2])")
plt.ylabel("y (uniform: [-2, 2])")
plt.grid()
plt.title("Sparse Grid Samples (Log-Uniform)")
plt.show()

# ============================================================================
# 2. Incremental sampling with QoI evaluation and active_keys
# ============================================================================
print("\n=== Test 2: Incremental Sampling with Sigmoid QoI ===")

all_parameters = {"x": 0, "y": 1, "unused": 0}


def sigmoid_qoi(active_param) -> np.ndarray:
    """QoI: [sum, sigmoid]"""
    for key, value in active_param.items():
        all_parameters[key] = value
    x = all_parameters["x"]
    y = all_parameters["y"]
    s = np.log(x) + y**2
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


sampler = sampler_cls(
    distributions=["log_uniform", "uniform"],
    bounds=[[np.exp(-2), np.exp(2.)], [-2., 2.]],
    active_keys=['x', 'y'],
    qoi_fn=sigmoid_qoi,
    DOE_type='QRS',
    seed=0
)

# First batch
sampler.sample(N=10)
X_before = sampler.X.copy()
Y_before = sampler.Y.copy()

print(f"First batch: {len(X_before)} samples")

# Incremental sampling
sampler.sample(N=10, as_additional_points=True)
print(f"After incremental: {len(sampler.X)} total samples")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# First batch
cvals = Y_before[:, 1]
ax.scatter(X_before[:, 0], X_before[:, 1], c=cvals, cmap="viridis", 
          s=100, marker='+', label='Batch 1', edgecolors='black', linewidths=1.5)

# Incremental batch
cvals_new = sampler.Y[len(X_before):, 1]
ax.scatter(sampler.X[len(X_before):, 0], sampler.X[len(X_before):, 1], 
          c=cvals_new, cmap="viridis", s=50, label='Batch 2 (incremental)')

ax.set_xlabel("x (log-uniform: [-2, 2])")
ax.set_ylabel("y (uniform: [-2, 2])")
ax.set_title("Incremental Sampling: QRS")
ax.grid(True, alpha=0.3)
ax.legend()
plt.colorbar(ax.collections[0], ax=ax, label='Sigmoid QoI')
plt.tight_layout()
plt.show()

# ============================================================================
# 3. Train metamodel and predict on grid
# ============================================================================
print("\n=== Test 3: Polynomial Model + Grid Prediction ===")

model = polynomial_lasso_regressor(order=4, coeff_reg=1e-4, SG=False)
model.train(sampler.X_reference, sampler.Y)

print(f"Model trained on {len(sampler.X)} samples")

# Test set
sampler_test = sampler_cls(
    distributions=["log_uniform", "uniform"],
    bounds=[[np.exp(-2), np.exp(2.)], [-2., 2.]],
    seed=1
)
sampler_test.sample(N=30)

preds = model.predict(sampler_test.X_reference)

# Visualize: training vs test predictions
fig, ax = plt.subplots(figsize=(8, 6))

# Training points (true QoI)
cvals_train = sampler.Y[:, 1]
sc_train = ax.scatter(sampler.X[:, 0], sampler.X[:, 1], c=cvals_train, 
                      cmap="viridis", s=80, label='Training (true)', 
                      edgecolors='black', linewidths=0.5)

# Test predictions
cvals_test = preds[:, 1]
ax.scatter(sampler_test.X[:, 0], sampler_test.X[:, 1], marker="x", 
          c=cvals_test, cmap="viridis", s=100, label='Test (predicted)', linewidths=2)

ax.set_xlabel("x (log-uniform: [-2, 2])")
ax.set_ylabel("y (uniform: [-2, 2])")
ax.set_title("Model Predictions on Test Set")
ax.grid(True, alpha=0.3)
ax.legend()
plt.colorbar(sc_train, ax=ax, label='Sigmoid QoI')
plt.tight_layout()
plt.show()

# ============================================================================
# 4. Contour plot on grid
# ============================================================================
print("\n=== Test 4: Contour Grid Prediction ===")

# Physical space grid
xmin, xmax = -1, 1
ymin, ymax = -1, 1
ng = 100

xx = np.linspace(xmin, xmax, ng)
yy = np.linspace(ymin, ymax, ng)
Xg, Yg = np.meshgrid(xx, yy)

grid_ref = np.c_[Xg.ravel(), Yg.ravel()]  # (ng*ng, 2) in physical space

# Convert to reference space

# Predict on grid
pred_grid = model.predict(grid_ref)  # shape (ng*ng, n_out)
Z = pred_grid[:, 1].reshape(ng, ng)  # QoI[1]

Xg, Yg = np.meshgrid(xx, yy)

U = np.c_[Xg.ravel(), Yg.ravel()]
U_phys = sampler.reference_to_physical(U)  # Normalize grid for prediction
Xg_phys = U_phys[:, 0].reshape(Xg.shape)          # (ny, nx)
Yg_phys = U_phys[:, 1].reshape(Yg.shape)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(Xg_phys, Yg_phys, Z, levels=30, cmap="viridis")
cbar = plt.colorbar(cf, ax=ax, label="Predicted Sigmoid QoI")

# Overlay training points
ax.scatter(sampler.X[:, 0], sampler.X[:, 1],
           c=sampler.Y[:, 1], cmap="viridis", 
           edgecolors="black", linewidths=0.5, s=40, 
           label="Training points", zorder=5)

ax.set_xlabel("x (log-uniform: [-2, 2])")
ax.set_ylabel("y (uniform: [-2, 2])")
ax.set_title("Polynomial Model Contour + Training Points")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

print("\n✓ Demo completed successfully!")
