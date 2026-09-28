"""
Core lognormal model utilities for quantile-surface experiments.

All synthetic: Z ~ N(mu, A), X_i = exp(Z_i).
# Custom — Context7 found no library equivalent for lognormal quantile-surface primitives
"""
import numpy as np
from numpy.linalg import inv, cholesky, LinAlgError
from typing import Dict, Any, Tuple

# Paper scenarios (B = precision matrix, A = covariance = B^{-1})
SCENARIOS: Dict[str, Dict[str, Any]] = {
    'S1': {
        'mu': np.array([0.0, 0.0]),
        'B': np.array([[0.5, 0.1], [0.1, 0.25]])
    },
    'S2': {
        'mu': np.array([0.0, 0.0]),
        'B': np.array([[5.0, 0.1], [0.1, 0.25]])
    },
    'S3': {
        'mu': np.array([0.0, 0.0]),
        'B': np.array([[0.5, -0.3], [-0.3, 0.25]])
    }
}


def validate_spd(matrix: np.ndarray, name: str = 'matrix') -> bool:
    """Check that matrix is symmetric and positive definite via Cholesky.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to validate.
    name : str
        Label for error messages.

    Returns
    -------
    bool
        True if SPD.

    Raises
    ------
    ValueError
        If matrix is not symmetric or not positive definite.
    """
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f'{name} is not symmetric')
    try:
        cholesky(matrix)
    except LinAlgError:
        raise ValueError(f'{name} is not positive definite')
    return True


def compute_A_B(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given scenario params with either B or A, compute both A=cov and B=precision.

    Parameters
    ----------
    params : dict
        Must contain 'mu' and either 'B' (precision) or 'A' (covariance).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mu, A, B) where A = B^{-1}.
    """
    mu = np.array(params['mu'], dtype=float)
    if 'B' in params:
        B = np.array(params['B'], dtype=float)
        validate_spd(B, 'B')
        A = inv(B)
    elif 'A' in params:
        A = np.array(params['A'], dtype=float)
        validate_spd(A, 'A')
        B = inv(A)
    else:
        raise ValueError('params must contain B (precision) or A (covariance)')
    validate_spd(A, 'A')
    validate_spd(B, 'B')
    return mu, A, B


def compute_b_beta(B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute b_i = sum_j B_{ji} (column sums) and beta = sum_i b_i.

    Parameters
    ----------
    B : np.ndarray
        Precision matrix.

    Returns
    -------
    Tuple[np.ndarray, float]
        (b, beta) where b[i] = sum_j B[j, i].
    """
    b = B.sum(axis=0)   # b_i = sum_j B_{ji} = column sum of B
    beta = float(b.sum())
    return b, beta


def classify_regime(B: np.ndarray, A: np.ndarray) -> str:
    """Classify scenario regime using the paper's sign and dominance conditions.

    In d=2:
      - barycentric : b_1 > 0 and b_2 > 0
      - left_corner : b_1 > 0 > b_2
      - right_corner: b_2 > 0 > b_1 (or both <= 0 → dominant corner)

    Parameters
    ----------
    B : np.ndarray
        Precision matrix.
    A : np.ndarray
        Covariance matrix (A = B^{-1}).

    Returns
    -------
    str
        Regime label.
    """
    b, _ = compute_b_beta(B)
    if len(b) != 2:
        return 'barycentric' if np.all(b > 0) else 'corner'
    b1, b2 = b[0], b[1]
    if b1 > 0 and b2 > 0:
        return 'barycentric'
    elif b1 > 0 and b2 <= 0:
        return 'left_corner'
    elif b2 > 0 and b1 <= 0:
        return 'right_corner'
    else:
        return 'right_corner'


def theoretical_optimizer(B: np.ndarray, A: np.ndarray) -> Dict[str, Any]:
    """Return theoretical simplex optimizer u* based on regime.

    Parameters
    ----------
    B : np.ndarray
        Precision matrix.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    dict
        Keys: 'regime', 'u_star', 't_star'.
    """
    b, beta = compute_b_beta(B)
    regime = classify_regime(B, A)
    if regime == 'barycentric':
        u_star = b / beta
        return {'regime': regime, 'u_star': u_star, 't_star': float(u_star[0])}
    elif regime == 'left_corner':
        return {'regime': regime, 'u_star': np.array([1.0, 0.0]), 't_star': 1.0}
    else:  # right_corner
        return {'regime': regime, 'u_star': np.array([0.0, 1.0]), 't_star': 0.0}


def generate_lognormal_samples(
    mu: np.ndarray,
    A: np.ndarray,
    N: int,
    seed: int = 42
) -> np.ndarray:
    """Generate N lognormal samples X_i = exp(Z_i), Z ~ N(mu, A).

    Parameters
    ----------
    mu : np.ndarray
        Mean vector of the Gaussian.
    A : np.ndarray
        Covariance matrix of the Gaussian.
    N : int
        Number of samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (N, d) array of lognormal samples.
    """
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(mu, A, size=N)
    return np.exp(Z)


def setup_scenario(name: str) -> Dict[str, Any]:
    """Set up a named scenario and return all derived quantities.

    Parameters
    ----------
    name : str
        Scenario name: 'S1', 'S2', or 'S3'.

    Returns
    -------
    dict
        Contains mu, A, B, b, beta, regime, optimizer.
    """
    params = SCENARIOS[name]
    mu, A, B = compute_A_B(params)
    b, beta = compute_b_beta(B)
    regime = classify_regime(B, A)
    opt = theoretical_optimizer(B, A)
    return {
        'name': name,
        'mu': mu,
        'A': A,
        'B': B,
        'b': b,
        'beta': beta,
        'regime': regime,
        'optimizer': opt,
    }
