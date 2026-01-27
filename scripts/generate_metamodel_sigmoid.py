import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surroptim.sampler import sampler_cls
from polynomial_meta_models import polynomial_ridge_regressor, polynomial_lasso_regressor
from surroptim.util import prediction_plot


def sigmoid_qoi(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[:, 0]
    y = X[:, 1]
    s = x + y**2
    sig = 1.0 / (1.0 + np.exp(-s))
    return np.vstack([s, sig]).T


def run_case(case_name: str, doe_type: str, N: int, model, grid_points_physical: np.ndarray, grid_points_norm: np.ndarray, Xg_phys: np.ndarray, Yg_phys: np.ndarray, use_normalised: bool = True) -> None:
    sampler = sampler_cls(
        distributions=["uniform", "uniform"],
        bounds=[[-2, 2], [-2, 2]],
        compute_QoIs=sigmoid_qoi,
        DOE_type=doe_type,
    )
    sampler.sampling(N=N)

    # Train on normalized or physical space
    X_train = sampler.X_normalised if use_normalised else sampler.X
    grid_points = grid_points_norm if use_normalised else grid_points_physical
    space_label = "normalized" if use_normalised else "physical"
    
    model.train(X_train, sampler.Y)

    # Extract sparsity info for lasso models (capture from training output)
    sparsity_info = ""
    if case_name == "lhs_lasso" and hasattr(model, 'coeff_'):
        coeff = model.coeff_
        if coeff.ndim == 2:
            coeff = coeff[0]
        n_nonzero = np.count_nonzero(coeff)
        n_total = coeff.size
        sparsity_ratio = (n_total - n_nonzero) / n_total
        sparsity_info = f"\nSparsity: {sparsity_ratio:.1%}"

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
    plt.savefig(f"plots/mm_sigmoid_samples_{case_name}.png", dpi=150)
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
    axes[1].set_title(f"{case_name} prediction\nR² = {r2:.4f}{sparsity_info}")
    axes[1].set_xlabel("x")

    plt.tight_layout()
    plt.savefig(f"plots/mm_sigmoid_prediction_{case_name}.png", dpi=150)
    plt.close()

    print(f"[OK] {case_name:15s} (R² = {r2:.4f}){sparsity_info}")
    print(f"     Samples: plots/mm_sigmoid_samples_{case_name}.png")
    print(f"     Predict: plots/mm_sigmoid_prediction_{case_name}.png\n")


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    # Parse command-line argument: --physical to use physical space, default is normalized
    use_normalised = "--physical" not in sys.argv
    
    print(f"Training space: {'normalized [-1,1]^2' if use_normalised else 'physical [-2,2]^2'}\n")

    # Precompute grids: physical [-2,2]^2 and normalized [-1,1]^2
    grid_lin_phys = np.linspace(-2, 2, 120)
    grid_lin_norm = np.linspace(-1, 1, 120)
    
    Xg_phys, Yg_phys = np.meshgrid(grid_lin_phys, grid_lin_phys)
    Xg_norm, Yg_norm = np.meshgrid(grid_lin_norm, grid_lin_norm)
    
    grid_points_physical = np.column_stack((Xg_phys.ravel(), Yg_phys.ravel()))
    grid_points_norm = np.column_stack((Xg_norm.ravel(), Yg_norm.ravel()))

    # QRS with ridge regression
    run_case(
        case_name="qrs_ridge",
        doe_type="QRS",
        N=30,
        model=polynomial_ridge_regressor(order=4, coeff_reg=1e-6, SG=False),
        grid_points_physical=grid_points_physical,
        grid_points_norm=grid_points_norm,
        Xg_phys=Xg_phys,
        Yg_phys=Yg_phys,
        use_normalised=use_normalised,
    )

    # LHS with sparse (lasso) regression
    run_case(
        case_name="lhs_lasso",
        doe_type="LHS",
        N=30,
        model=polynomial_lasso_regressor(order=4, coeff_reg=1e-3, SG=False, use_sklearn=False),
        grid_points_physical=grid_points_physical,
        grid_points_norm=grid_points_norm,
        Xg_phys=Xg_phys,
        Yg_phys=Yg_phys,
        use_normalised=use_normalised,
    )

    # Sparse grid sampling with ridge regression (refinement level as N)
    run_case(
        case_name="sg_ridge",
        doe_type="SG",
        N=4,
        model=polynomial_ridge_regressor(order=4, coeff_reg=1e-6, SG=True),
        grid_points_physical=grid_points_physical,
        grid_points_norm=grid_points_norm,
        Xg_phys=Xg_phys,
        Yg_phys=Yg_phys,
        use_normalised=use_normalised,
    )


if __name__ == "__main__":
    main()
