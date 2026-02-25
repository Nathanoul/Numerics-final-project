import numpy as np


def thomas_solver(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d.

    Parameters:
    a : array-like
        The sub-diagonal (elements below the main diagonal).
        Length should be N-1. (a[0] is below b[1])
    b : array-like
        The main diagonal. Length N.
    c : array-like
        The super-diagonal (elements above the main diagonal).
        Length should be N-1. (c[0] is above b[0])
    d : array-like
        The right-hand side vector. Length N.

    Returns:
    x : ndarray
        The solution vector of length N.
    """
    n = len(d)
    # Use float64 to prevent precision loss with large k jumps
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)

    # 1. Forward Elimination phase
    # First row
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    # Intermediate rows
    for i in range(1, n - 1):
        denominator = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denominator
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denominator

    # Last row
    denominator = b[n - 1] - a[n - 2] * cp[n - 2]
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / denominator

    # 2. Backward Substitution phase
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x