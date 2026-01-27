import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surroptim.sparse_grid import (
    clenshaw_curtis_compute, generate_sparse_grid, compute_n_from_level,
    generate_delta_grid, generate_multi_index, generate_list_orders_dim
)


def main():
    # Test Clenshaw-Curtis quadrature
    print("=== Clenshaw-Curtis Quadrature ===")
    
    for n in [3, 5, 9]:
        x, w = clenshaw_curtis_compute(n)
        print(f"\nOrder {n}:")
        print(f"  Nodes: {x}")
        print(f"  Weights: {w}")
        print(f"  Sum of weights: {np.sum(w)}")
    
    # Test compute_n_from_level
    print("\n=== Refinement Level to n ===")
    print(f"compute_n_from_level(3): {compute_n_from_level(3)}")
    print(f"compute_n_from_level(4): {compute_n_from_level(4)}")
    
    # Test generate_delta_grid
    print("\n=== Delta Grid ===")
    print(f"generate_delta_grid(2): {generate_delta_grid(2)}")
    print(f"generate_delta_grid(3): {generate_delta_grid(3)}")
    
    # Test generate_multi_index
    print("\n=== Multi-Index ===")
    print(f"generate_multi_index(2, 3): {generate_multi_index(2, 3)}")
    
    # Test 2D meshgrid
    print("\n=== 2D Meshgrid Test ===")
    x, _ = clenshaw_curtis_compute(3)
    y, _ = clenshaw_curtis_compute(5)
    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    print(f"Meshgrid points (3x5): shape {points.shape}")
    print(f"Points:\n{points}")
    
    # Test sparse grid generation
    print("\n=== Sparse Grid Generation ===")
    dim = 2
    N = 5  # number of refinement levels in hierarchical grid
    sg_points = generate_sparse_grid(dim, N)
    print(f"Sparse grid (dim={dim}, N={N}): {len(sg_points)} points")
    
    # Visualize 2D sparse grid
    if dim == 2:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(sg_points[:, 0], sg_points[:, 1])
        plt.grid()
        plt.title(f"Sparse Grid (dim={dim}, N={N})")
        
        # Also plot full tensor product for comparison
        x, _ = clenshaw_curtis_compute(compute_n_from_level(5))
        full_points = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
        plt.subplot(1, 2, 2)
        plt.scatter(full_points[:, 0], full_points[:, 1], alpha=0.5)
        plt.grid()
        plt.title(f"Full Tensor Grid (level 5)")
        
        plt.tight_layout()
        plt.show()
    
    # Test 3D sparse grid
    if False:  # Set to True to visualize 3D
        dim = 3
        N = 3
        sg_points_3d = generate_sparse_grid(dim, N)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sg_points_3d[:, 0], sg_points_3d[:, 1], sg_points_3d[:, 2])
        
        # Highlight slice
        filtered_arr = sg_points_3d[(sg_points_3d[:, 0] >= -0.01) & (sg_points_3d[:, 0] <= 0.01)]
        ax.scatter(filtered_arr[:, 0], filtered_arr[:, 1], filtered_arr[:, 2], c='black', alpha=1)
        plt.show()
    
    # Test list_orders_dim
    print("\n=== List Orders Dim ===")
    dim = 2
    N = 5
    list_orders_dim = generate_list_orders_dim(dim, N)
    print(f"list_orders_dim(dim={dim}, N={N}): {list_orders_dim}")
    print(f"  Length: {len(list_orders_dim)}")
    print(f"  Sparse grid points: {len(sg_points)}")


if __name__ == "__main__":
    main()
