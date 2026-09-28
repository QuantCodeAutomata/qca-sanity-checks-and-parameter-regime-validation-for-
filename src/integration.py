"""
Deterministic 2D integration utilities for lognormal quantile computation.

Provides exact CDF evaluation and quantile inversion for bivariate lognormal portfolios.
# Custom — Context7 found no library equivalent for paper's 2D integration scheme
"""
import numpy as np
from numpy.linalg import inv
from scipy import integrate, optimize
from scipy.stats import lognorm
from typing import Tuple


def lognormal_portfolio_pdf_levelset(
    y: float,
    t: float,
    mu: np.ndarray,
    B: np.ndarray,
) -> float:
    """Compute f_Y(y) for Y = t*X_1 + (1-t)*X_2 via level-set integration.

    f_Y(y) = ∫₀^{y/t} f_X(x1, (y - t*x1)/(1-t)) / (1-t) dx1

    Parameters
    ----------
    y : float
        PDF evaluation point.
    t : float
        Weight on X_1.
    mu : np.ndarray
        Gaussian mean.
    B : np.ndarray
        Precision matrix.

    Returns
    -------
    float
        PDF value f_Y(y).
    """
    if y <= 0.0:
        return 0.0
    if t <= 1e-10 or t >= 1.0 - 1e-10:
        return 0.0

    A = inv(B)
    det_A = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    if det_A <= 0.0:
        return 0.0
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_A))

    # Clip to effective X_1 support to keep quad's domain manageable
    sigma_1 = np.sqrt(A[0, 0])
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / t, x1_eff_max)

    def integrand(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return np.exp(-phi) * norm_const / (x1 * x2 * (1.0 - t))

    result, _ = integrate.quad(
        integrand, 1e-15, x1_max,
        limit=300, epsabs=1e-10, epsrel=1e-10,
    )
    return max(float(result), 0.0)


def lognormal_portfolio_cdf(
    y: float,
    t: float,
    mu: np.ndarray,
    B: np.ndarray,
) -> float:
    """Compute P(t*X_1 + (1-t)*X_2 <= y) via conditional distribution 1D integral.

    Uses the factorisation:
      F_Y(y) = ∫₀^{y/t} f_{X_1}(x1) · Φ((log((y-t·x1)/(1-t)) - cond_mu(x1)) / cond_std) dx1
    where the conditional is from the bivariate Gaussian structure.

    Parameters
    ----------
    y : float
        Portfolio value threshold.
    t : float
        Weight on X_1 (weight on X_2 is 1-t).
    mu : np.ndarray
        Mean vector (length 2).
    B : np.ndarray
        Precision matrix (2x2).

    Returns
    -------
    float
        CDF value in [0, 1].
    """
    from scipy.stats import norm as _norm

    if y <= 0.0:
        return 0.0

    A = inv(B)

    # Degenerate cases: pure single-asset portfolios
    if t <= 1e-10:
        return float(lognorm.cdf(y, s=np.sqrt(A[1, 1]), scale=np.exp(mu[1])))
    if t >= 1.0 - 1e-10:
        return float(lognorm.cdf(y, s=np.sqrt(A[0, 0]), scale=np.exp(mu[0])))

    A00, A01, A11 = A[0, 0], A[0, 1], A[1, 1]
    # Conditional log-X2 | log-X1 = lx1 ~ N(cond_mu, cond_var)
    cond_var = A11 - A01 ** 2 / A00
    if cond_var <= 0.0:
        return np.nan
    cond_std = np.sqrt(cond_var)

    def integrand(x1: float) -> float:
        if x1 <= 0.0:
            return 0.0
        x2_max = (y - t * x1) / (1.0 - t)
        if x2_max <= 0.0:
            return 0.0
        # Marginal PDF of X_1 (lognormal)
        f_x1 = float(lognorm.pdf(x1, s=np.sqrt(A00), scale=np.exp(mu[0])))
        # Conditional CDF of X_2 | X_1 = x1 evaluated at x2_max
        lx1 = np.log(x1)
        cond_mean_lx2 = mu[1] + A01 / A00 * (lx1 - mu[0])
        cond_cdf = float(_norm.cdf((np.log(x2_max) - cond_mean_lx2) / cond_std))
        return f_x1 * cond_cdf

    # Clip integration upper bound to the effective support of X_1:
    # beyond exp(mu[0] + 7*sigma_1) the lognormal density is essentially zero,
    # so extending the range further only causes quad to fail on a huge domain.
    sigma_1 = np.sqrt(A00)
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / t, x1_eff_max)

    result, _ = integrate.quad(
        integrand, 1e-15, x1_max,
        limit=300, epsabs=1e-9, epsrel=1e-9,
    )
    return float(np.clip(result, 0.0, 1.0))


def lognormal_portfolio_quantile_det(
    alpha: float,
    t: float,
    mu: np.ndarray,
    B: np.ndarray,
    y_bounds: Tuple[float, float] = (1e-10, 1e6)
) -> float:
    """Compute the quantile q_alpha(t) deterministically via CDF inversion.

    Parameters
    ----------
    alpha : float
        Quantile level.
    t : float
        Weight on X_1.
    mu : np.ndarray
        Mean vector.
    B : np.ndarray
        Precision matrix.
    y_bounds : tuple
        Bracket (lower, upper) for root finding.

    Returns
    -------
    float
        Quantile value.
    """
    def f(y: float) -> float:
        return lognormal_portfolio_cdf(y, t, mu, B) - alpha

    try:
        return float(optimize.brentq(f, y_bounds[0], y_bounds[1],
                                     xtol=1e-10, rtol=1e-10))
    except ValueError:
        return np.nan


def lognormal_cond_expectation_x(
    y: float,
    t: float,
    mu: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """Compute E[X | t*X_1 + (1-t)*X_2 = y] by integration on the level set.

    This validates the gradient formula: nabla_u q_alpha(u) = E[X | u^T X = q_alpha].

    Parameters
    ----------
    y : float
        Conditioning level.
    t : float
        Weight on X_1.
    mu : np.ndarray
        Mean vector.
    B : np.ndarray
        Precision matrix.

    Returns
    -------
    np.ndarray
        Conditional expectation vector [E[X_1|...], E[X_2|...]].
    """
    A = inv(B)
    det_A = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    if det_A <= 0.0:
        return np.array([np.nan, np.nan])
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_A))

    if t <= 1e-10 or t >= 1.0 - 1e-10:
        return np.array([np.nan, np.nan])

    # Clip to effective X_1 support (same as in CDF)
    sigma_1 = np.sqrt(A[0, 0])
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / t, x1_eff_max)

    def density_level_set(x1: float) -> float:
        """Density on level set (unnormalized), proportional to f_X(x)/||u||."""
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        # f_X / ||u|| where ||u|| = sqrt(t^2 + (1-t)^2)
        return np.exp(-phi) * norm_const / (x1 * x2)

    Z_y, _ = integrate.quad(density_level_set, 1e-15, x1_max, limit=300)

    if Z_y < 1e-300:
        return np.array([np.nan, np.nan])

    def integ_x1(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return x1 * np.exp(-phi) * norm_const / (x1 * x2)

    def integ_x2(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return x2 * np.exp(-phi) * norm_const / (x1 * x2)

    E_x1, _ = integrate.quad(integ_x1, 1e-15, x1_max, limit=300)
    E_x2, _ = integrate.quad(integ_x2, 1e-15, x1_max, limit=300)

    return np.array([E_x1 / Z_y, E_x2 / Z_y])


def lognormal_cond_var_x(
    y: float,
    t: float,
    mu: np.ndarray,
    B: np.ndarray
) -> float:
    """Compute Var(v^T X | u^T X = y) for v = u = (t, 1-t).

    Parameters
    ----------
    y : float
        Conditioning level.
    t : float
        Weight on X_1.
    mu : np.ndarray
        Mean vector.
    B : np.ndarray
        Precision matrix.

    Returns
    -------
    float
        Conditional variance.
    """
    A = inv(B)
    det_A = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    if det_A <= 0.0 or t <= 1e-10 or t >= 1.0 - 1e-10:
        return np.nan
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_A))

    # Clip to effective X_1 support (same as in CDF)
    sigma_1 = np.sqrt(A[0, 0])
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / t, x1_eff_max)

    def density_ls(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return np.exp(-phi) * norm_const / (x1 * x2)

    Z_y, _ = integrate.quad(density_ls, 1e-15, x1_max, limit=300)
    if Z_y < 1e-300:
        return np.nan

    # v^T X = t*X_1 + (1-t)*X_2 = y on the level set (constant!)
    # So Var(v^T X | u^T X = y) = 0 when v = u.
    # For a different direction, say v = e_1:
    def integ_portval(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        # v = e_1: x_1
        return x1 * np.exp(-phi) * norm_const / (x1 * x2)

    def integ_portval2(x1: float) -> float:
        x2 = (y - t * x1) / (1.0 - t)
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return (x1 ** 2) * np.exp(-phi) * norm_const / (x1 * x2)

    E_x1, _ = integrate.quad(integ_portval, 1e-15, x1_max, limit=300)
    E_x12, _ = integrate.quad(integ_portval2, 1e-15, x1_max, limit=300)
    E_x1 /= Z_y
    E_x12 /= Z_y
    return max(E_x12 - E_x1 ** 2, 0.0)


# ---------------------------------------------------------------------------
# General two-weight (unconstrained u1, u2 > 0) versions for Theorem 1.1 tests
# ---------------------------------------------------------------------------

def lognormal_portfolio_cdf_gen(
    y: float,
    u1: float,
    u2: float,
    mu: np.ndarray,
    B: np.ndarray,
) -> float:
    """Compute P(u1*X_1 + u2*X_2 <= y) with independent weights u1, u2 > 0.

    Uses the same conditional-factorisation as lognormal_portfolio_cdf but
    without the simplex constraint u2 = 1 - u1.

    Parameters
    ----------
    y  : float  — portfolio threshold.
    u1 : float  — weight on X_1 (strictly positive).
    u2 : float  — weight on X_2 (strictly positive).
    mu : mean vector; B : precision matrix.
    """
    from scipy.stats import norm as _norm

    if y <= 0.0 or u1 <= 0.0 or u2 <= 0.0:
        return 0.0

    A = inv(B)
    A00, A01, A11 = A[0, 0], A[0, 1], A[1, 1]
    cond_var = A11 - A01 ** 2 / A00
    if cond_var <= 0.0:
        return np.nan
    cond_std = np.sqrt(cond_var)

    sigma_1 = np.sqrt(A00)
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / u1, x1_eff_max)

    def integrand(x1: float) -> float:
        if x1 <= 0.0:
            return 0.0
        x2_max = (y - u1 * x1) / u2
        if x2_max <= 0.0:
            return 0.0
        f_x1 = float(lognorm.pdf(x1, s=np.sqrt(A00), scale=np.exp(mu[0])))
        lx1 = np.log(x1)
        cond_mean_lx2 = mu[1] + A01 / A00 * (lx1 - mu[0])
        return f_x1 * float(_norm.cdf((np.log(x2_max) - cond_mean_lx2) / cond_std))

    result, _ = integrate.quad(
        integrand, 1e-15, x1_max, limit=300, epsabs=1e-9, epsrel=1e-9
    )
    return float(np.clip(result, 0.0, 1.0))


def lognormal_portfolio_quantile_gen(
    alpha: float,
    u1: float,
    u2: float,
    mu: np.ndarray,
    B: np.ndarray,
    y_bounds: Tuple[float, float] = (1e-10, 1e6),
) -> float:
    """Quantile q_alpha for Y = u1*X_1 + u2*X_2 with independent weights."""
    def f(y: float) -> float:
        return lognormal_portfolio_cdf_gen(y, u1, u2, mu, B) - alpha
    try:
        return float(optimize.brentq(f, y_bounds[0], y_bounds[1],
                                     xtol=1e-10, rtol=1e-10))
    except ValueError:
        return np.nan


def lognormal_cond_expectation_gen(
    y: float,
    u1: float,
    u2: float,
    mu: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Compute E[X | u1*X_1 + u2*X_2 = y] with independent weights u1, u2 > 0."""
    A = inv(B)
    det_A = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    if det_A <= 0.0 or u1 <= 1e-10 or u2 <= 1e-10:
        return np.array([np.nan, np.nan])
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_A))

    sigma_1 = np.sqrt(A[0, 0])
    x1_eff_max = np.exp(mu[0] + 7.0 * sigma_1)
    x1_max = min(y / u1, x1_eff_max)

    def density_ls(x1: float) -> float:
        x2 = (y - u1 * x1) / u2
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return np.exp(-phi) * norm_const / (x1 * x2)

    Z_y, _ = integrate.quad(density_ls, 1e-15, x1_max, limit=300)
    if Z_y < 1e-300:
        return np.array([np.nan, np.nan])

    def integ_xi(x1: float, i: int) -> float:
        x2 = (y - u1 * x1) / u2
        if x1 <= 0.0 or x2 <= 0.0:
            return 0.0
        xi = x1 if i == 0 else x2
        log_x = np.array([np.log(x1), np.log(x2)])
        diff = log_x - mu
        phi = 0.5 * float(diff @ B @ diff)
        return xi * np.exp(-phi) * norm_const / (x1 * x2)

    E_x1, _ = integrate.quad(lambda x: integ_xi(x, 0), 1e-15, x1_max, limit=300)
    E_x2, _ = integrate.quad(lambda x: integ_xi(x, 1), 1e-15, x1_max, limit=300)
    return np.array([E_x1 / Z_y, E_x2 / Z_y])


# ---------------------------------------------------------------------------
# High-level wrappers used by Experiment 9
# ---------------------------------------------------------------------------

def quantile_lognormal_deterministic(
    u: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
    y_bounds: Tuple[float, float] = (1e-10, 1e6),
) -> float:
    """Compute q_alpha(u) for lognormal by deterministic CDF inversion.

    Uses the full unconstrained two-weight form Y = u[0]*X_1 + u[1]*X_2
    so that finite-difference gradients in each direction are consistent
    with the conditional-expectation formula ∇_u q_α = E[X | u^T X = q_α].

    Parameters
    ----------
    u : np.ndarray
        Weight vector [u1, u2] (both > 0, not required to sum to 1).
    alpha : float
        Quantile level.
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.
    y_bounds : tuple
        Bracket for root finding.

    Returns
    -------
    float
        Quantile estimate.
    """
    B = inv(A)
    return lognormal_portfolio_quantile_gen(alpha, float(u[0]), float(u[1]), mu, B, y_bounds)


def pdf_Y_lognormal(
    y: float,
    u: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
) -> float:
    """Estimate f_Y(y) for Y = u^T X via direct level-set integration.

    Parameters
    ----------
    y : float
        Query point.
    u : np.ndarray
        Weight vector.
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    float
        PDF value f_Y(y).
    """
    B = inv(A)
    t = float(u[0])
    return lognormal_portfolio_pdf_levelset(y, t, mu, B)


def conditional_expectation_surface(
    y: float,
    u: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Compute gradient formula E[X | u^T X = y] via level-set integration.

    Uses the general two-weight form Y = u[0]*X_1 + u[1]*X_2 so that the
    result is directly comparable to a finite-difference gradient computed by
    perturbing each weight independently.

    Parameters
    ----------
    y : float
        Conditioning level.
    u : np.ndarray
        Weight vector [u1, u2] (both > 0).
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Conditional expectation [E[X_1|...], E[X_2|...]].
    """
    B = inv(A)
    return lognormal_cond_expectation_gen(y, float(u[0]), float(u[1]), mu, B)


def conditional_variance_surface(
    y: float,
    u: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
) -> float:
    """Compute Var(v^T X | u^T X = y) via level-set integration.

    On the level set {u1 x1 + u2 x2 = y} we have x2 = (y − u1 x1)/u2,
    so v^T X = (v1 − v2 u1/u2) x1 + v2 y/u2.  Hence

        Var(v^T X | u^T X = y)  =  (v1 − v2 u1/u2)^2 * Var(X1 | u^T X = y)

    Parameters
    ----------
    y : float
        Conditioning level.
    u : np.ndarray
        Weight vector [u1, u2] (both > 0).
    v : np.ndarray
        Direction vector (not required to satisfy u^T v = 0).
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    float
        Var(v^T X | u^T X = y).
    """
    u1, u2 = float(u[0]), float(u[1])
    v1, v2 = float(v[0]), float(v[1])
    B = inv(A)
    # Var(X1 | u^T X = y) — uses simplex form internally but works for any t
    # mapping: use u1/sum as t for the existing helper
    t = u1 / (u1 + u2)
    # Scale y to the unit-sum parameterisation
    y_scaled = y / (u1 + u2)
    var_x1 = lognormal_cond_var_x(y_scaled, t, mu, B)
    # Linear coefficient of X1 in v^T X on the level set
    A_v = v1 - v2 * u1 / u2
    return A_v ** 2 * var_x1


def gaussian_quantile_analytic(
    u: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
) -> float:
    """Compute exact Gaussian quantile q_alpha(u) = u^T mu + z_alpha * sqrt(u^T A u).

    Parameters
    ----------
    u : np.ndarray
        Weight vector.
    alpha : float
        Quantile level.
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    float
        Analytic quantile.
    """
    from scipy.stats import norm
    z_alpha = float(norm.ppf(alpha))
    m = float(u @ mu)
    s = float(np.sqrt(u @ A @ u))
    return m + z_alpha * s


def gaussian_quantile_gradient_analytic(
    u: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Compute analytic gradient of Gaussian quantile with respect to u.

    d q_alpha / d u_i = mu_i + z_alpha * (A u)_i / sqrt(u^T A u)

    Parameters
    ----------
    u : np.ndarray
        Weight vector.
    alpha : float
        Quantile level.
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Gradient vector.
    """
    from scipy.stats import norm
    z_alpha = float(norm.ppf(alpha))
    s = float(np.sqrt(u @ A @ u))
    Au = A @ u
    return mu + z_alpha * Au / s


def gaussian_quantile_hessian_vv_analytic(
    u: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
) -> float:
    """Compute analytic directional second derivative D²q_alpha[v,v] for Gaussian.

    D²q[v,v] = z_alpha * [v^T A v / s - (u^T A v)^2 / s^3]
    where s = sqrt(u^T A u).

    Parameters
    ----------
    u : np.ndarray
        Weight vector.
    v : np.ndarray
        Direction vector.
    alpha : float
        Quantile level.
    mu : np.ndarray
        Gaussian mean.
    A : np.ndarray
        Covariance matrix.

    Returns
    -------
    float
        Analytic second derivative.
    """
    from scipy.stats import norm
    z_alpha = float(norm.ppf(alpha))
    s = float(np.sqrt(u @ A @ u))
    vAv = float(v @ A @ v)
    uAv = float(u @ A @ v)
    return z_alpha * (vAv / s - uAv ** 2 / s ** 3)
