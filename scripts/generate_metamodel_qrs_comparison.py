"""
Generate metamodel comparison plots for sigmoid QoI using QRS sampling with three MM strategies:
- Gaussian Process (GP)
- Neural Network (NN)
- k-Nearest Neighbors (k-NN)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surroptim.sampler import sampler_cls
from gaussian_process_meta_model import GP_regressor

try:
    from neural_network_meta_model import neural_net_regressor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, skipping Neural Network case")

from neighrest_neighbour_meta_model import NNeigh_regressor
from surroptim.util import prediction_plot


def sigmoid_qoi(X: np.ndarray) -> np.ndarray:
    """Sigmoid QoI: s = x + y^2, sig = sigmoid(s)."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[:, 0]
    y = X[:, 1]
    s = x + y**2
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


def run_case(
    case_name: str,
    model,
    N_samples: int,
    grid_points_physical: np.ndarray,
    grid_points_norm: np.ndarray,
    Xg_phys: np.ndarray,
    Yg_phys: np.ndarray,
    use_normalised: bool = True,
) -> None:
    """
    Run a single metamodel case with QRS sampling.

    Args:
        case_name: Name of the case (e.g., 'gp', 'nn', 'knn')
        model: Metamodel regressor instance
        N_samples: Number of QRS samples
        grid_points_physical: Grid points in physical space for plotting
        grid_points_norm: Grid points in normalized space for predictions
        Xg_phys: Physical space X meshgrid
        Yg_phys: Physical space Y meshgrid
        use_normalised: Whether to train in normalized space
    """
    # Generate QRS samples
    sampler = sampler_cls(
        distributions=["uniform", "uniform"],
        bounds=[[-2, 2], [-2, 2]],
        qoi_fn=sigmoid_qoi,
        DOE_type="QRS",
    )
    sampler.sample(N=N_samples)

    # Select training space
    X_train = sampler.X_normalised if use_normalised else sampler.X
    grid_points = grid_points_norm if use_normalised else grid_points_physical
    space_label = "normalized" if use_normalised else "physical"

    # Train model
    print(f"  Training {case_name.upper()} in {space_label} space...")
    model.train(X_train, sampler.Y)

    # Plot training samples (colored by sigmoid output) in physical space
    prediction_plot(
        X=sampler.X,
        y=sampler.Y[:, 1],
        xlabel="x",
        ylabel="y",
        clabel="sigmoid(x + y^2)",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"plots/mm_sigmoid_samples_qrs_{case_name}.png", dpi=150)
    plt.close()

    # Predict on selected grid
    preds = model.predict(grid_points)
    Z_pred = preds[:, 1].reshape(Xg_phys.shape)
    Z_true = sigmoid_qoi(grid_points_physical)[:, 1].reshape(Xg_phys.shape)

    # Compute R²
    ss_res = np.sum((Z_true.ravel() - Z_pred.ravel()) ** 2)
    ss_tot = np.sum((Z_true.ravel() - np.mean(Z_true.ravel())) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)

    # Plot on physical space
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    cs0 = axes[0].contourf(Xg_phys, Yg_phys, Z_true, levels=30, cmap="viridis")
    fig.colorbar(cs0, ax=axes[0], label="true sigmoid")
    axes[0].scatter(sampler.X[:, 0], sampler.X[:, 1], s=30, alpha=0.6, c='red', edgecolors='k', linewidth=0.5)
    axes[0].set_title("True sigmoid(x + y^2)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    cs1 = axes[1].contourf(Xg_phys, Yg_phys, Z_pred, levels=30, cmap="viridis")
    fig.colorbar(cs1, ax=axes[1], label="predicted")
    axes[1].scatter(sampler.X[:, 0], sampler.X[:, 1], s=30, alpha=0.6, c='red', edgecolors='k', linewidth=0.5)
    axes[1].set_title(f"{case_name.upper()} prediction\nR² = {r2:.4f}")
    axes[1].set_xlabel("x")

    plt.tight_layout()
    plt.savefig(f"plots/mm_sigmoid_prediction_qrs_{case_name}.png", dpi=150)
    plt.close()

    print(f"  [OK] {case_name.upper():6s} R² = {r2:.4f}")


def main() -> None:
    """Generate all metamodel comparison cases."""
    os.makedirs("plots", exist_ok=True)

    # Parse command-line argument
    use_normalised = "--physical" not in sys.argv
    space_label = "normalized [-1,1]^2" if use_normalised else "physical [-2,2]^2"
    
    print(f"\n{'='*60}")
    print(f"QRS Metamodel Comparison: {space_label}")
    print(f"{'='*60}\n")

    # Precompute grids
    grid_lin_phys = np.linspace(-2, 2, 120)
    grid_lin_norm = np.linspace(-1, 1, 120)
    
    Xg_phys, Yg_phys = np.meshgrid(grid_lin_phys, grid_lin_phys)
    Xg_norm, Yg_norm = np.meshgrid(grid_lin_norm, grid_lin_norm)
    
    grid_points_physical = np.column_stack((Xg_phys.ravel(), Yg_phys.ravel()))
    grid_points_norm = np.column_stack((Xg_norm.ravel(), Yg_norm.ravel()))

    N_samples = 30

    # Gaussian Process
    print(f"Case 1: Gaussian Process (N={N_samples} QRS samples)")
    run_case(
        case_name="gp",
        model=GP_regressor(length_scale=1.0, length_scale_bounds=(0.1, 10.0)),
        N_samples=N_samples,
        grid_points_physical=grid_points_physical,
        grid_points_norm=grid_points_norm,
        Xg_phys=Xg_phys,
        Yg_phys=Yg_phys,
        use_normalised=use_normalised,
    )

    # Neural Network (only if torch available)
    if HAS_TORCH:
        print(f"Case 2: Neural Network (N={N_samples} QRS samples)")
        run_case(
            case_name="nn",
            model=neural_net_regressor(n_hidden=50),
            N_samples=N_samples,
            grid_points_physical=grid_points_physical,
            grid_points_norm=grid_points_norm,
            Xg_phys=Xg_phys,
            Yg_phys=Yg_phys,
            use_normalised=use_normalised,
        )
    else:
        print(f"Case 2: Neural Network - SKIPPED (PyTorch not installed)\n")

    # k-Nearest Neighbors
    print(f"Case 3: k-Nearest Neighbors (N={N_samples} QRS samples)")
    run_case(
        case_name="knn",
        model=NNeigh_regressor(n_neighbors=5, weights='distance'),
        N_samples=N_samples,
        grid_points_physical=grid_points_physical,
        grid_points_norm=grid_points_norm,
        Xg_phys=Xg_phys,
        Yg_phys=Yg_phys,
        use_normalised=use_normalised,
    )

    print(f"{'='*60}")
    print("All cases completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
