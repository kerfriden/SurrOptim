import numpy as np
import itertools

try:
    from config import SPARSE_GRID_TOLERANCE, SPARSE_GRID_MAX_REFINEMENT_LEVEL
except ImportError:  # When imported as a package
    from .config import SPARSE_GRID_TOLERANCE, SPARSE_GRID_MAX_REFINEMENT_LEVEL

#*****************************************************************************80
#
## CLENSHAW_CURTIS_COMPUTE computes a Clenshaw Curtis quadrature rule.
#
#  Discussion:
#
#    This method uses a direct approach.  The paper by Waldvogel
#    exhibits a more efficient approach using Fourier transforms.
#
#    The integral:
#
#      integral ( -1 <= x <= 1 ) f(x) dx
#
#    The quadrature rule:
#
#      sum ( 1 <= i <= n ) w(i) * f ( x(i) )
#
#    The abscissas for the rule of order N can be regarded
#    as the cosines of equally spaced angles between 180 and 0 degrees:
#
#      X(I) = cos ( ( N - I ) * PI / ( N - 1 ) )
#
#    except for the basic case N = 1, when
#
#      X(1) = 0.
#
#    A Clenshaw-Curtis rule that uses N points will integrate
#    exactly all polynomials of degrees 0 through N-1.  If N
#    is odd, then by symmetry the polynomial of degree N will
#    also be integrated exactly.
#
#    If the value of N is increased in a sensible way, then
#    the new set of abscissas will include the old ones.  One such
#    sequence would be N(K) = 2*K+1 for K = 0, 1, 2, ...
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    02 April 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Charles Clenshaw, Alan Curtis,
#    A Method for Numerical Integration on an Automatic Computer,
#    Numerische Mathematik,
#    Volume 2, Number 1, December 1960, pages 197-205.
#
#    Philip Davis, Philip Rabinowitz,
#    Methods of Numerical Integration,
#    Second Edition,
#    Dover, 2007,
#    ISBN: 0486453391,
#    LC: QA299.3.D28.
#
#    Joerg Waldvogel,
#    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules,
#    BIT Numerical Mathematics,
#    Volume 43, Number 1, 2003, pages 1-18.
#
#  Parameters:
#
#    Input, integer N, the order.
#
#    Output, real X(N), the abscissas.
#
#    Output, real W(N), the weights.
#
def clenshaw_curtis_compute(n: int):
    if ( n == 1 ):

        x = np.zeros ( n )
        w = np.zeros ( n )

        w[0] = 2.0

    else:

        theta = np.zeros ( n )

        for i in range ( 0, n ):
            theta[i] = float ( n - 1 - i ) * np.pi / float ( n - 1 )

        x = np.cos ( theta )
        w = np.zeros ( n )

        for i in range ( 0, n ):

            w[i] = 1.0

            jhi = ( ( n - 1 ) // 2 )

            for j in range ( 0, jhi ):

                if ( 2 * ( j + 1 ) == ( n - 1 ) ):
                    b = 1.0
                else:
                    b = 2.0

                w[i] = w[i] - b * np.cos ( 2.0 * float ( j + 1 ) * theta[i] ) \
                     / float ( 4 * j * ( j + 2 ) + 3 )

        w[0] = w[0] / float ( n - 1 )
        for i in range ( 1, n - 1 ):
            w[i] = 2.0 * w[i] / float ( n - 1 )
        w[n-1] = w[n-1] / float ( n - 1 )

    return x, w


def compute_n_from_level(l: int) -> int:
    """
    Map refinement level to number of quadrature nodes.

    Args:
        l: Refinement level (0, 1, 2, ...)

    Returns:
        Number of nodes

    Raises:
        ValueError: If level is invalid
    """
    if l < 0:
        raise ValueError(f"Refinement level must be non-negative, got {l}")

    if l == 0:
        return 0
    elif l == 1:
        return 1
    elif l == 2:
        return 3
    else:
        n = 3
        for _ in range(l - 2):
            n = n + (n - 1)
        return n

def generate_delta_grid(N: int) -> list:
    """
    Generate hierarchical delta grids for sparse grid construction.

    Args:
        N: Number of refinement levels

    Returns:
        List of point arrays for each level

    Raises:
        ValueError: If N exceeds maximum refinement level
    """
    if N > SPARSE_GRID_MAX_REFINEMENT_LEVEL:
        raise ValueError(
            f"Refinement level {N} exceeds maximum {SPARSE_GRID_MAX_REFINEMENT_LEVEL}"
        )

    list_delta_x = [clenshaw_curtis_compute(1)[0].tolist()]
    
    for i in range(N - 1):
        delta_x_prev, _ = clenshaw_curtis_compute(compute_n_from_level(i + 1))
        delta_x, _ = clenshaw_curtis_compute(compute_n_from_level(i + 2))
        mask = np.all(np.abs(delta_x[:, None] - delta_x_prev) > SPARSE_GRID_TOLERANCE, axis=1)
        delta_x = delta_x[mask]
        list_delta_x.append(delta_x.tolist())
    
    return list_delta_x


import itertools

def generate_multi_index(D: int, N: int) -> list:
    """
    Generate multi-index for sparse grid combinations.

    Args:
        D: Number of dimensions
        N: Maximum sum of indices

    Returns:
        List of multi-index tuples with sum <= N

    Raises:
        ValueError: If D or N are invalid
    """
    if D <= 0:
        raise ValueError(f"Dimension must be positive, got {D}")
    if N < 0:
        raise ValueError(f"Maximum sum must be non-negative, got {N}")

    ranges = [range(N + 1) for _ in range(D)]
    multiindex = [index for index in itertools.product(*ranges) if sum(index) <= N]

    return multiindex


def meshgrid_from_list(vectors: list) -> np.ndarray:
    """
    Generate N-dimensional points from coordinate vectors.

    Args:
        vectors: List of 1D arrays for each dimension

    Returns:
        Array of shape (n_points, N) with all coordinate combinations

    Raises:
        ValueError: If vectors list is empty
    """
    if not vectors:
        raise ValueError("Vectors list cannot be empty")

    grids = np.meshgrid(*vectors, indexing='ij')
    grid_flat = [g.ravel() for g in grids]
    return np.vstack(grid_flat).T

def generate_sparse_grid(dim: int, N: int) -> np.ndarray:
    """
    Generate sparse grid points using Clenshaw-Curtis quadrature.

    Args:
        dim: Problem dimensionality
        N: Refinement level (number of hierarchical levels)

    Returns:
        Array of shape (n_points, dim) with sparse grid points in [-1, 1]^dim

    Raises:
        ValueError: If dim or N are invalid
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if N <= 0:
        raise ValueError(f"Refinement level must be positive, got {N}")

    MI = generate_multi_index(dim, N - 1)
    list_delta_x = generate_delta_grid(N)
    points = []
    
    for i in range(len(MI)):
        vectors = [list_delta_x[j] for j in MI[i]]
        points_i = meshgrid_from_list(vectors)
        points.append(points_i.tolist())
    
    points = [item for sublist in points for item in sublist]
    return np.array(points)


from itertools import product

def generate_integers(i: int, j: int) -> list:
    """Generate list of integers from i to j-1."""
    if i >= j:
        raise ValueError(f"Start ({i}) must be less than end ({j})")
    return list(range(i, j))

def generate_list_orders(N: int) -> list:
    """Generate list of order ranges for each refinement level."""
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    list_orders = []
    for i in range(N):
        n1 = compute_n_from_level(i)
        n2 = compute_n_from_level(i + 1)
        list_orders.append(generate_integers(n1, n2))
    return list_orders

def generate_combinations(list_orders: list, index_vectors: list) -> list:
    """Generate all combinations of orders for given indices."""
    if not list_orders:
        raise ValueError("list_orders cannot be empty")
    if not index_vectors:
        raise ValueError("index_vectors cannot be empty")

    all_combinations = []
    for index_vector in index_vectors:
        selected_sublists = [list_orders[i] for i in index_vector]
        all_combinations.extend(itertools.product(*selected_sublists))
    return all_combinations

def generate_list_orders_dim(dim: int, N: int) -> list:
    """Generate list of orders for all dimensions."""
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    list_orders = generate_list_orders(N)
    MI = generate_multi_index(dim, N - 1)
    list_orders_dim = generate_combinations(list_orders, MI)
    return list_orders_dim