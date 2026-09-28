"""
Regime classification, theoretical optimizer, and normalized shape utilities.

# Custom — Context7 found no library equivalent for paper regime classification
"""
import numpy as np
from typing import Dict, Any
from src.model import compute_b_beta


def verify_dominance_equivalence(B: np.ndarray, A: np.ndarray) -> Dict[str, Any]:
    """Verify that b_1^2/B_11 > b_2^2/B_22 is numerically equivalent to A_22 > A_11 in 2D.

    Parameters
    ----------
    B : np.ndarray
        2x2 precision matrix.
    A : np.ndarray
        2x2 covariance matrix.

    Returns
    -------
    dict
        Comparison results with both sides of the equivalence.
    """
    b, _ = compute_b_beta(B)
    b1sq_B11 = (b[0] ** 2) / B[0, 0]
    b2sq_B22 = (b[1] ** 2) / B[1, 1]
    lhs = bool(b1sq_B11 > b2sq_B22)     # b_1^2/B_11 > b_2^2/B_22
    rhs = bool(A[1, 1] > A[0, 0])       # A_22 > A_11
    return {
        'lhs': lhs,
        'rhs': rhs,
        'equivalent': lhs == rhs,
        'b1sq_over_B11': b1sq_B11,
        'b2sq_over_B22': b2sq_B22,
        'A11': A[0, 0],
        'A22': A[1, 1],
    }


def theoretical_normalized_shape_barycentric(
    t_grid: np.ndarray,
    b: np.ndarray,
    beta: float
) -> np.ndarray:
    """Compute theoretical normalized shape q_0(t) for the 2D barycentric regime.

    q_0(t) = exp((1/beta) * [b_1 * log(beta*t/b_1) + b_2 * log(beta*(1-t)/b_2)])

    Valid for t in (0, 1); endpoint singularities are clipped.

    Parameters
    ----------
    t_grid : np.ndarray
        Grid values in [0, 1].
    b : np.ndarray
        b vector (length 2), both components positive.
    beta : float
        beta = sum(b).

    Returns
    -------
    np.ndarray
        Theoretical shape at each t.
    """
    b1, b2 = b[0], b[1]
    eps = 1e-12
    t = np.clip(t_grid, eps, 1.0 - eps)
    log_term = b1 * np.log(beta * t / b1) + b2 * np.log(beta * (1.0 - t) / b2)
    return np.exp(log_term / beta)


def section5_optimizer_formula(A: np.ndarray) -> np.ndarray:
    """Compute Section 5 optimizer formula in terms of A for the 2D barycentric case.

    u_1 = (A_22 - A_12) / (A_11 + A_22 - 2*A_12)
    u_2 = (A_11 - A_12) / (A_11 + A_22 - 2*A_12)

    Parameters
    ----------
    A : np.ndarray
        2x2 covariance matrix.

    Returns
    -------
    np.ndarray
        Optimizer [u_1, u_2].
    """
    A11, A22, A12 = A[0, 0], A[1, 1], A[0, 1]
    denom = A11 + A22 - 2.0 * A12
    u1 = (A22 - A12) / denom
    u2 = (A11 - A12) / denom
    return np.array([u1, u2])
