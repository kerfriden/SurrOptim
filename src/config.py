"""
Configuration constants for the SurrOptim package.

Centralizes all hardcoded values and configuration parameters.
"""

# Sparse grid constants
SPARSE_GRID_TOLERANCE = 1.0e-6
SPARSE_GRID_MAX_REFINEMENT_LEVEL = 10

# QRS (Sobol) constants
QRS_SOBOL_M = 20  # 2^20 points = 1,048,576 points

# Numerical constants
NUMERICAL_EPSILON = 1.0e-12

# Distribution types
SUPPORTED_DISTRIBUTIONS = ['uniform', 'log_uniform']

# DOE types
SUPPORTED_DOE_TYPES = ['PRS', 'LHS', 'QRS', 'SG']
