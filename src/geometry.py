"""
Surface-measure geometry: level-set pullback for lognormal distributions.

Implements the paper's w_y, g_y, Z_y, and induced measure mu_y on S_1.
# Custom — Context7 found no library equivalent for paper's surface pullback construction
"""
import numpy as np
from scipy import integrate, optimize
from typing import Tuple, Dict


def phi_X(x1: float, x2: float, mu: np.ndarray, B: np.ndarray) -> float:
    """Negative log of lognormal density (up to constant).

    phi_X(x) = 0.5*(log x - mu)^T B (log x - mu) + sum_j log x_j

    Parameters
    ----------
    x1, x2 : float
        Positive coordinates.
    mu : np.ndarray
        Gaussian mean (length 2).
    B : np.ndarray
        Precision matrix (2x2).

    Returns
    -------
    float
        phi_X value; +inf if x1 or x2 <= 0.
    """
    if x1 <= 0.0 or x2 <= 0.0:
        return np.inf
    log_x = np.array([np.log(x1), np.log(x2)])
    diff = log_x - mu
    return 0.5 * float(diff @ B @ diff) + np.log(x1) + np.log(x2)


def compute_a_scalar(x1: float, x2: float, b: np.ndarray) -> float:
    """Compute a(x) = sum_i b_i * log(x_i).

    Parameters
    ----------
    x1, x2 : float
        Positive coordinates.
    b : np.ndarray
        b vector.

    Returns
    -------
    float
    """
    return b[0] * np.log(x1) + b[1] * np.log(x2)


def w_y(
    x1: float,
    u: np.ndarray,
    y: float,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray
) -> float:
    """Exponent w_y(x) = phi_X(x) - log(y)*a(x) on the level set u^T x = y.

    g_y(x) = exp(-w_y(x)) ∝ y^{a(x)} * f_X(x).
    Parameterized by x1; x2 = (y - u[0]*x1)/u[1].

    Parameters
    ----------
    x1 : float
        First coordinate.
    u : np.ndarray
        Portfolio weight vector (length 2, positive entries).
    y : float
        Level set value.
    mu : np.ndarray
        Gaussian mean.
    B : np.ndarray
        Precision matrix.
    b : np.ndarray
        b vector.

    Returns
    -------
    float
        w_y value; +inf if outside domain.
    """
    x2 = (y - u[0] * x1) / u[1]
    if x1 <= 0.0 or x2 <= 0.0:
        return np.inf
    phi = phi_X(x1, x2, mu, B)
    a_val = compute_a_scalar(x1, x2, b)
    return phi - np.log(y) * a_val


def g_y(
    x1: float,
    u: np.ndarray,
    y: float,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray
) -> float:
    """Un-normalized density g_y(x1) = exp(-w_y(x1)) on level set.

    Returns 0 outside the domain.
    """
    val = w_y(x1, u, y, mu, B, b)
    if np.isinf(val) or np.isnan(val):
        return 0.0
    # Prevent overflow for extreme values
    if val > 700.0:
        return 0.0
    return float(np.exp(-val))


def compute_Z_y(
    u: np.ndarray,
    y: float,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray
) -> Tuple[float, float]:
    """Compute normalization Z_y = integral of g_y over level set S_1.

    In 2D, the level set {x > 0 : u^T x = y} is parameterized by
    x1 in (0, y/u[0]).

    Parameters
    ----------
    u : np.ndarray
        Portfolio weights (positive, sum <= 1 each).
    y : float
        Level value.
    mu, B, b : arrays
        Model parameters.

    Returns
    -------
    Tuple[float, float]
        (Z_y, integration_error).
    """
    x1_max = y / u[0]
    result = integrate.quad(
        lambda x1: g_y(x1, u, y, mu, B, b),
        1e-12,
        x1_max * (1.0 - 1e-10),
        limit=300,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return result[0], result[1]


def compute_mode_x1(
    u: np.ndarray,
    y: float,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray
) -> float:
    """Find mode x1* = argmin w_y(x1) on the level set (argmax of g_y).

    Parameters
    ----------
    u : np.ndarray
        Portfolio weights.
    y : float
        Level value.
    mu, B, b : arrays
        Model parameters.

    Returns
    -------
    float
        x1 value at the mode.
    """
    x1_max = y / u[0]
    result = optimize.minimize_scalar(
        lambda x1: w_y(x1, u, y, mu, B, b),
        bounds=(1e-10, x1_max * (1.0 - 1e-10)),
        method='bounded',
        options={'xatol': 1e-12},
    )
    return float(result.x)


def compute_moments_mu_y(
    u: np.ndarray,
    y: float,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray
) -> Dict[str, float]:
    """Compute moments of the induced measure mu_y on S_1.

    Computes E_{mu_y}[a], E_{mu_y}[x_1] (= E_{mu_y}[gamma] for v=(1,0)),
    and Var_{mu_y}[x_1].

    Parameters
    ----------
    u : np.ndarray
        Portfolio weights (length 2, positive).
    y : float
        Level value.
    mu, B, b : arrays
        Model parameters.

    Returns
    -------
    dict
        Keys: 'E_a', 'E_gamma', 'Var_gamma', 'Z_y'.
    """
    x1_max = y / u[0]
    upper = x1_max * (1.0 - 1e-10)
    lower = 1e-12

    def base(x1: float) -> float:
        return g_y(x1, u, y, mu, B, b)

    Z, _ = integrate.quad(base, lower, upper, limit=300, epsabs=1e-10)

    if Z < 1e-300:
        return {'E_a': np.nan, 'E_gamma': np.nan, 'Var_gamma': np.nan, 'Z_y': Z}

    def integ_a(x1: float) -> float:
        x2 = (y - u[0] * x1) / u[1]
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        return compute_a_scalar(x1, x2, b) * g_y(x1, u, y, mu, B, b)

    E_a, _ = integrate.quad(integ_a, lower, upper, limit=300, epsabs=1e-10)
    E_a /= Z

    def integ_g1(x1: float) -> float:
        return x1 * g_y(x1, u, y, mu, B, b)

    E_g, _ = integrate.quad(integ_g1, lower, upper, limit=300, epsabs=1e-10)
    E_g /= Z

    def integ_g2(x1: float) -> float:
        return x1 ** 2 * g_y(x1, u, y, mu, B, b)

    E_g2, _ = integrate.quad(integ_g2, lower, upper, limit=300, epsabs=1e-10)
    E_g2 /= Z
    Var_g = E_g2 - E_g ** 2

    return {'E_a': E_a, 'E_gamma': E_g, 'Var_gamma': max(Var_g, 0.0), 'Z_y': Z}


# ---------------------------------------------------------------------------
# Vectorized and aliased interfaces for Experiment 6
# ---------------------------------------------------------------------------

def lognormal_log_density_phi(
    x1: np.ndarray,
    x2: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Vectorized negative log-density phi_X for lognormal.

    phi_X(x) = 0.5*(log x - mu)^T B (log x - mu) + sum log x_j

    Parameters
    ----------
    x1, x2 : np.ndarray
        Positive coordinate arrays (same shape).
    mu : np.ndarray
        Gaussian mean.
    B : np.ndarray
        Precision matrix.

    Returns
    -------
    np.ndarray
        phi_X values; +inf where x1 or x2 <= 0.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    result = np.full_like(x1, np.inf)
    valid = (x1 > 0.0) & (x2 > 0.0)
    if not np.any(valid):
        return result
    lx1 = np.log(x1[valid])
    lx2 = np.log(x2[valid])
    diff0 = lx1 - mu[0]
    diff1 = lx2 - mu[1]
    qf = (B[0, 0] * diff0 ** 2 + 2.0 * B[0, 1] * diff0 * diff1 + B[1, 1] * diff1 ** 2)
    result[valid] = 0.5 * qf + lx1 + lx2
    return result


def compute_g_y(
    y: float,
    x1_eval: np.ndarray,
    u: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
    log_scale: bool = False,
) -> np.ndarray:
    """Vectorized g_y on the level set for an array of x1 values.

    For each x1, x2 = (y - u[0]*x1) / u[1].
    g_y(x1) = exp(-w_y(x1)) = exp(-phi_X(x) + log(y)*a(x)).

    Parameters
    ----------
    y : float
        Level set value.
    x1_eval : np.ndarray
        Array of x1 values.
    u : np.ndarray
        Portfolio weights (positive, length 2).
    mu, B, b : arrays
        Model parameters.
    log_scale : bool
        If True, return log(g_y) instead of g_y.

    Returns
    -------
    np.ndarray
        g_y values (or log g_y if log_scale=True).
    """
    x1 = np.asarray(x1_eval, dtype=float)
    x2 = (y - u[0] * x1) / u[1]
    valid = (x1 > 0.0) & (x2 > 0.0)

    phi = lognormal_log_density_phi(x1, x2, mu, B)
    # a(x) = b[0]*log(x1) + b[1]*log(x2)
    log_x1 = np.where(x1 > 0.0, np.log(np.where(x1 > 0.0, x1, 1.0)), -np.inf)
    log_x2 = np.where(x2 > 0.0, np.log(np.where(x2 > 0.0, x2, 1.0)), -np.inf)
    a_vals = b[0] * log_x1 + b[1] * log_x2

    log_gy = np.where(valid, np.log(y) * a_vals - phi, -np.inf)
    if log_scale:
        return log_gy
    # Stabilize using log-sum-exp shift
    log_gy = np.where(np.isfinite(log_gy), log_gy, -np.inf)
    return np.where(valid, np.exp(log_gy), 0.0)


def compute_Zy_normalization(
    y: float,
    u: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float]:
    """Compute normalization Z_y (alias for compute_Z_y).

    Parameters
    ----------
    y : float
        Level set value.
    u, mu, B, b : arrays
        Model parameters.

    Returns
    -------
    Tuple[float, float]
        (Z_y, integration_error).
    """
    return compute_Z_y(u, y, mu, B, b)


def compute_mu_y_moments(
    y: float,
    x1_eval: np.ndarray,
    u: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
) -> Dict[str, float]:
    """Compute moments of induced measure mu_y using integration.

    Wraps compute_moments_mu_y, ignoring x1_eval (uses integration internally).

    Parameters
    ----------
    y : float
        Level value.
    x1_eval : np.ndarray
        Evaluation grid (not used; kept for API consistency with exp6).
    u, mu, B, b : arrays
        Model parameters.

    Returns
    -------
    dict
        {'E_a', 'E_gamma', 'Var_gamma', 'Z_y'}.
    """
    return compute_moments_mu_y(u, y, mu, B, b)


def compute_mode_gy(
    y: float,
    u: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
    x1_lo: float,
    x1_hi: float,
) -> float:
    """Find mode of g_y on level set with explicit bounds.

    Parameters
    ----------
    y : float
        Level value.
    u, mu, B, b : arrays
        Model parameters.
    x1_lo : float
        Lower bound for x1.
    x1_hi : float
        Upper bound for x1.

    Returns
    -------
    float
        x1 at mode.
    """
    lo = max(x1_lo, 1e-12)
    hi = min(x1_hi, y / u[0] * (1.0 - 1e-10))
    if lo >= hi:
        return np.nan
    result = optimize.minimize_scalar(
        lambda x1: w_y(x1, u, y, mu, B, b),
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': 1e-10},
    )
    return float(result.x)


def compute_projected_hessian_criterion(
    y: float,
    u: np.ndarray,
    mu: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
    n_x1: int = 500,
    delta: float = 0.02,
) -> float:
    """Evaluate sign criterion for projected Hessian via finite difference.

    Computes the numerical derivative of [f_Y(y) * Var_{mu_y}(X_1)]
    and returns it as a proxy for the Hessian sign (negative => concave).

    Numerically: d/dy [Z_y * Var_{mu_y}(gamma)] using central differences.

    Parameters
    ----------
    y : float
        Level value.
    u, mu, B, b : arrays
        Model parameters.
    n_x1 : int
        Grid resolution for vectorized computations.
    delta : float
        Relative step for finite difference.

    Returns
    -------
    float
        Derivative of [Z_y * Var_gamma] at y; sign indicates curvature.
    """
    h = delta * y

    def moment_at(yy: float) -> float:
        mom = compute_moments_mu_y(u, yy, mu, B, b)
        return mom['Z_y'] * mom['Var_gamma']

    try:
        val_plus = moment_at(y + h)
        val_minus = moment_at(y - h)
        if np.isnan(val_plus) or np.isnan(val_minus):
            return np.nan
        return (val_plus - val_minus) / (2.0 * h)
    except Exception:
        return np.nan
