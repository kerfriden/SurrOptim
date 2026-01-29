#!/usr/bin/env python3
"""Export Clenshaw-Curtis quadrature point images: sparse vs tensor (full) grids.

Generates and saves PNG images under the repository `plots/` folder.

Usage:
    python scripts/export_quadrature_images.py --level 4 --dim 2
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from surroptim.sparse_grid import (
    generate_sparse_grid,
    clenshaw_curtis_compute,
    compute_n_from_level,
    meshgrid_from_list,
)


def plot_points(points: np.ndarray, ax, label=None, color="C0", s=20):
    pts = np.asarray(points)
    if pts.size == 0:
        return
    if pts.ndim == 1:
        x = pts
        y = np.zeros_like(x)
    else:
        x = pts[:, 0]
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros_like(x)
    ax.scatter(x, y, s=s, label=label, alpha=0.8, color=color)


def export_plots(level: int = 4, dim: int = 2, out_dir: str | None = None) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if out_dir is None:
        out_dir = os.path.join(repo_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Sparse grid points (in [-1,1]^dim)
    sparse_pts = generate_sparse_grid(dim, level)

    # Tensor (full) grid: use Clenshaw-Curtis nodes per dimension
    n_nodes = compute_n_from_level(level)
    nodes, _ = clenshaw_curtis_compute(n_nodes)
    vectors = [nodes.tolist() for _ in range(dim)]
    tensor_pts = meshgrid_from_list(vectors)

    # Prepare plotting: if dim==1, plot points along x-axis; if dim>2 project first two dims
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    plot_points(sparse_pts, ax, label=f"Sparse L{level}", color="C1", s=18)
    ax.set_title(f"Sparse grid (Clenshaw–Curtis) — level {level}")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", "box")
    ax.legend()

    ax = axes[1]
    plot_points(tensor_pts, ax, label=f"Tensor (full), nodes/dim={n_nodes}", color="C0", s=18)
    ax.set_title(f"Tensor (full) Clenshaw–Curtis — nodes/dim {n_nodes}")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", "box")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"quadrature_level_{level}_dim{dim}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Export sparse vs tensor Clenshaw-Curtis quadrature plots")
    p.add_argument("--level", type=int, default=4, help="Clenshaw-Curtis refinement level (integer)")
    p.add_argument("--dim", type=int, default=2, help="Problem dimension (2 recommended for plotting)")
    p.add_argument("--out", type=str, default=None, help="Output directory (defaults to repo 'plots/')")
    args = p.parse_args()

    path = export_plots(level=args.level, dim=args.dim, out_dir=args.out)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
