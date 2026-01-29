#!/usr/bin/env python3
"""Plot 2D tensor-product Legendre basis functions for several index sets.

Generates PNGs under `plots/` for:
 - total-order (order=2)
 - full tensor-product (order=2)
 - sparse-grid index set (level=2)

Usage: python -m surroptim.scripts.plot_basis_functions
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from surroptim.polynomials import (
    Legendre,
    poly_basis_multi_index,
    generate_multi_index,
    generate_tensor_product_index,
)
from surroptim.sparse_grid import generate_list_orders_dim


def plot_basis_grid(MI, title, out_path, grid_n=201):
    """Arrange basis functions on a grid: cols = order_x, rows = order_y.

    MI is a list of multi-index tuples (ix, iy). Missing cells are left blank.
    """
    # derive max orders
    orders = np.asarray(MI)
    if orders.ndim == 1:
        # single-dimension entries (unlikely) -> treat as list of tuples
        orders = orders.reshape(-1, 1)
    max_x = int(np.max(orders[:, 0]))
    max_y = int(np.max(orders[:, 1]))

    xs = np.linspace(-1, 1, grid_n)
    Ys, Xs = np.meshgrid(xs, xs)
    pts = np.vstack((Xs.ravel(), Ys.ravel())).T

    A = poly_basis_multi_index(pts, Legendre, MI)

    # create mapping from multi-index -> column index in A
    mi_to_col = {tuple(mi): j for j, mi in enumerate(MI)}

    ncols = max_x + 1
    nrows = max_y + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    # axes indexed [row, col] with row=iy, col=ix; flip y-axis so higher order up
    for iy in range(nrows):
        for ix in range(ncols):
            ax = axes[iy, ix] if nrows > 1 and ncols > 1 else (axes[max(iy, ix)] if (nrows > 1 or ncols > 1) else axes)
            mi = (ix, iy)
            if mi in mi_to_col:
                col = mi_to_col[mi]
                vals = A[:, col].reshape((grid_n, grid_n))
                # force constant basis (0,0) to display on [0,1]; leave others auto-scaled
                if mi == (0, 0):
                    im = ax.contourf(xs, xs, vals.T, levels=40, cmap="RdBu_r", vmin=0.0, vmax=2.0)
                else:
                    im = ax.contourf(xs, xs, vals.T, levels=40, cmap="RdBu_r")
                ax.set_title(f"{mi}")
                # Use fixed formatting for colorbar and hide offset text to avoid
                # tiny scientific-offset annotations like "1e-14 + 1".
                cbar = fig.colorbar(im, ax=ax, shrink=0.8, format="%.2f")
                try:
                    cbar.ax.yaxis.get_offset_text().set_visible(False)
                except Exception:
                    pass
            else:
                ax.axis("off")
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    order = 2
    level = 2
    dim = 2

    # total-order MI (sum <= order)
    MI_total = generate_multi_index(dim, order)
    out_total = os.path.join(os.path.dirname(__file__), "..", "..", "plots", f"basis_total_order_{order}_2d.png")
    out_total = os.path.abspath(out_total)
    plot_basis_grid(MI_total, f"Total-order (order={order})", out_total)
    print("Saved:", out_total)

    # tensor-product full MI
    MI_tensor = generate_tensor_product_index(dim, order)
    out_tensor = os.path.join(os.path.dirname(__file__), "..", "..", "plots", f"basis_tensor_order_{order}_2d.png")
    out_tensor = os.path.abspath(out_tensor)
    plot_basis_grid(MI_tensor, f"Tensor-product (order={order})", out_tensor)
    print("Saved:", out_tensor)

    # sparse-grid MI via generate_list_orders_dim (uses level)
    MI_sparse = generate_list_orders_dim(dim, level)
    out_sparse = os.path.join(os.path.dirname(__file__), "..", "..", "plots", f"basis_sparse_level_{level}_2d.png")
    out_sparse = os.path.abspath(out_sparse)
    plot_basis_grid(MI_sparse, f"Sparse-grid MI (level={level})", out_sparse)
    print("Saved:", out_sparse)


if __name__ == "__main__":
    main()
