import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surroptim.sampler import sampler_old_cls as sampler_cls
from surroptim.util import prediction_plot


def rosen_qoi(X: np.ndarray) -> np.ndarray:
    """Two-output QoI: sum and sigmoid(x + y**2) for a curved response."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x, y = X[:, 0], X[:, 1]
    s = x + y
    sig = 1.0 / (1.0 + np.exp(-(x + y**2)))
    return np.vstack([s, sig]).T


def main() -> None:
    os.makedirs("plots", exist_ok=True)
    # Note: for doe_type='SG', the sampler interprets N as sparse-grid refinement level.
    # In 2D, level=5 yields 65 points (close to 64) for a fair visual comparison.
    cases = [("QRS", 64), ("LHS", 64), ("PRS", 64), ("SG", 5)]

    for doe, N in cases:
        sampler = sampler_cls(
            distributions=["uniform", "uniform"],
            bounds=[[-2, 2], [-2, 2]],
            qoi_fn=rosen_qoi,
            DOE_type=doe,
        )
        sampler.sample(N=N)
        prediction_plot(
            X=sampler.X,
            y=sampler.Y[:, 1],
            xlabel="x",
            ylabel="y",
            clabel="sigmoid(x + y^2)",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(f"plots/rosen_{doe.lower()}.png", dpi=150)
        plt.close()

    print(
        "Saved plots to plots/rosen_qrs.png, rosen_lhs.png, rosen_prs.png, rosen_sg.png"
    )


if __name__ == "__main__":
    main()
