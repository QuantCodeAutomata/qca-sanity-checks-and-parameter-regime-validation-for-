"""
Empirical quantile utilities implementing the paper's inf-based left-continuous definition.

# Custom — Context7 found no library equivalent for the paper's inf-based quantile rule
"""
import numpy as np
from typing import Dict, List


def empirical_quantile_left(Y: np.ndarray, alpha: float) -> float:
    """Left-continuous empirical quantile (inf-definition).

    Select index ceil(alpha * N) - 1 using np.partition (O(N)) for speed.

    Parameters
    ----------
    Y : np.ndarray
        1-D sample array.
    alpha : float
        Quantile level in (0, 1].

    Returns
    -------
    float
        Empirical quantile.
    """
    N = len(Y)
    if N == 0:
        raise ValueError('Y is empty')
    idx = int(np.clip(int(np.ceil(alpha * N)) - 1, 0, N - 1))
    return float(np.partition(Y, idx)[idx])


def empirical_quantile_curve(
    X: np.ndarray,
    t_grid: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute empirical quantile curve q_alpha(t) for u=(t, 1-t) over t_grid.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 2) lognormal samples.
    t_grid : np.ndarray
        Grid of t values in [0, 1].
    alpha : float
        Quantile level.

    Returns
    -------
    np.ndarray
        Shape (len(t_grid),) array of quantiles.
    """
    q = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        Y = t * X[:, 0] + (1.0 - t) * X[:, 1]
        q[i] = empirical_quantile_left(Y, alpha)
    return q


def empirical_quantile_multi_alpha(
    X: np.ndarray,
    t_grid: np.ndarray,
    alphas: List[float]
) -> Dict[float, np.ndarray]:
    """Compute empirical quantile curves for multiple alpha values.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 2) lognormal samples.
    t_grid : np.ndarray
        Grid of t values in [0, 1].
    alphas : list of float
        Quantile levels.

    Returns
    -------
    dict
        {alpha: q_curve array}.
    """
    return {alpha: empirical_quantile_curve(X, t_grid, alpha) for alpha in alphas}


def compute_curvature(q: np.ndarray, dt: float) -> np.ndarray:
    """Estimate d^2 q / dt^2 via centered finite differences.

    Parameters
    ----------
    q : np.ndarray
        Quantile curve of length M.
    dt : float
        Grid spacing.

    Returns
    -------
    np.ndarray
        Length M-2 curvature array (interior points only).
    """
    return (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)


def gaussian_quantile_on_simplex(
    A: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    t_grid: np.ndarray
) -> np.ndarray:
    """Exact Gaussian quantile curve q_alpha(t) = m(t) + z_alpha * s(t).

    For Y_t = t*X_1 + (1-t)*X_2 with X ~ N(mu, A):
      m(t) = u^T mu, s(t) = sqrt(u^T A u), z_alpha = Phi^{-1}(alpha).

    # Using scipy.stats.norm — Context7 confirmed

    Parameters
    ----------
    A : np.ndarray
        Covariance matrix (2x2).
    mu : np.ndarray
        Mean vector (length 2).
    alpha : float
        Quantile level.
    t_grid : np.ndarray
        Grid of t values.

    Returns
    -------
    np.ndarray
        Analytic quantile curve.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(alpha)
    q = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        u = np.array([t, 1.0 - t])
        m = float(u @ mu)
        s = float(np.sqrt(u @ A @ u))
        q[i] = m + z_alpha * s
    return q
