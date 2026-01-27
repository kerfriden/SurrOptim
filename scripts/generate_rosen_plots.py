import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surroptim.sampler import sampler_cls
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
    cases = [("QRS", 64), ("LHS", 64), ("PRS", 64), ("SG", 3)]

    for doe, N in cases:
        sampler = sampler_cls(
            distributions=["uniform", "uniform"],
            bounds=[[-2, 2], [-2, 2]],
            compute_QoIs=rosen_qoi,
            n_out=2,
        )
        sampler.sampling(N=N)
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
