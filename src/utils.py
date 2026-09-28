"""
Shared utility functions for quantile-surface experiments.
"""
import numpy as np
from typing import Optional, Dict, List


def simplex_grid(n_points: int = 201) -> np.ndarray:
    """Generate a dense simplex grid t in [0, 1].

    Parameters
    ----------
    n_points : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        Evenly spaced values in [0, 1].
    """
    return np.linspace(0.0, 1.0, n_points)


def normalized_shape(
    q_curve: np.ndarray,
    t_grid: np.ndarray,
    k_alpha: Optional[float] = None
) -> np.ndarray:
    """Normalize quantile curve by its maximum value (or provided k_alpha).

    Parameters
    ----------
    q_curve : np.ndarray
        Quantile curve array.
    t_grid : np.ndarray
        Grid (unused here, kept for API consistency).
    k_alpha : float, optional
        Normalization constant; defaults to max(q_curve).

    Returns
    -------
    np.ndarray
        Normalized curve.
    """
    if k_alpha is None:
        k_alpha = float(np.max(q_curve))
    if k_alpha == 0.0:
        return np.zeros_like(q_curve)
    return q_curve / k_alpha


def find_optimizer(q_curve: np.ndarray, t_grid: np.ndarray) -> float:
    """Find t* = argmax_t q_curve(t).

    Parameters
    ----------
    q_curve : np.ndarray
        Quantile values.
    t_grid : np.ndarray
        Corresponding t values.

    Returns
    -------
    float
        t value at the maximum.
    """
    idx = int(np.argmax(q_curve))
    return float(t_grid[idx])


def shape_error_metrics(
    q_norm: np.ndarray,
    q_theory: np.ndarray,
    t_grid: np.ndarray,
    eps: float = 1e-4
) -> Dict[str, float]:
    """Compute sup-norm and supremum relative error on the interior grid.

    Parameters
    ----------
    q_norm : np.ndarray
        Empirical normalized shape.
    q_theory : np.ndarray
        Theoretical shape.
    t_grid : np.ndarray
        Grid values.
    eps : float
        Interior margin (evaluates on t in [eps, 1-eps]).

    Returns
    -------
    dict
        {'sup_norm': ..., 'sup_rel': ...}
    """
    mask = (t_grid >= eps) & (t_grid <= 1.0 - eps)
    diff = np.abs(q_norm[mask] - q_theory[mask])
    sup_norm = float(np.max(diff))
    denom = np.abs(q_theory[mask]) + 1e-15
    sup_rel = float(np.max(diff / denom))
    return {'sup_norm': sup_norm, 'sup_rel': sup_rel}


def check_homogeneity(
    X: np.ndarray,
    u: np.ndarray,
    alpha: float,
    c_values: List[float],
    tol: float = 0.05
) -> Dict[float, Dict]:
    """Test positive homogeneity q_alpha(c*u) ≈ c * q_alpha(u).

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 2) lognormal samples.
    u : np.ndarray
        Weight vector.
    alpha : float
        Quantile level.
    c_values : list of float
        Scalar multipliers to test.
    tol : float
        Relative error tolerance for pass/fail.

    Returns
    -------
    dict
        {c: {'q_cu', 'c_q_u', 'rel_err', 'passes'}}.
    """
    from src.quantile import empirical_quantile_left
    Y_u = X @ u
    q_u = empirical_quantile_left(Y_u, alpha)
    results: Dict[float, Dict] = {}
    for c in c_values:
        Y_cu = X @ (c * u)
        q_cu = empirical_quantile_left(Y_cu, alpha)
        expected = c * q_u
        rel_err = abs(q_cu - expected) / (abs(expected) + 1e-15)
        results[c] = {
            'q_cu': q_cu,
            'c_q_u': expected,
            'rel_err': rel_err,
            'passes': bool(rel_err < tol),
        }
    return results
