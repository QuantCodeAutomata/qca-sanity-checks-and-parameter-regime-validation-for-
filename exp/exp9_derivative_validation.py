"""
Experiment 9: Direct numerical validation of quantile gradient and Hessian
formulas (Theorem 1.1) using finite differences and deterministic 2D integration.

Validates:
  ∇_u q_α(u) = E[X | u^T X = q_α(u)]
  D² q_α(u)[v,v] = -f_Y(y,u)^{-1} * d/dy [ f_Y(y,u) Var(v^T X | u^T X = y) ] @ y=q
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import setup_scenario, generate_lognormal_samples
from src.quantile import empirical_quantile_left
from src.integration import (
    quantile_lognormal_deterministic,
    conditional_expectation_surface,
    conditional_variance_surface,
    pdf_Y_lognormal,
    gaussian_quantile_analytic,
    gaussian_quantile_gradient_analytic,
    gaussian_quantile_hessian_vv_analytic,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def finite_diff_gradient(
    u: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
    epsilon: float = 0.005,
    use_gaussian: bool = False,
) -> np.ndarray:
    """Estimate gradient of q_alpha(u) by central finite differences.

    Parameters
    ----------
    u : Portfolio weight vector.
    alpha : Quantile level.
    mu, A : Model parameters.
    epsilon : Step size.
    use_gaussian : If True compute Gaussian analytic quantile.

    Returns
    -------
    np.ndarray: Gradient estimate.
    """
    d = len(u)
    grad = np.zeros(d)
    e = np.eye(d)
    for i in range(d):
        u_plus = u + epsilon * e[i]
        u_minus = u - epsilon * e[i]
        if use_gaussian:
            q_plus = gaussian_quantile_analytic(u_plus, alpha, mu, A)
            q_minus = gaussian_quantile_analytic(u_minus, alpha, mu, A)
        else:
            q_plus = quantile_lognormal_deterministic(u_plus, alpha, mu, A)
            q_minus = quantile_lognormal_deterministic(u_minus, alpha, mu, A)
        grad[i] = (q_plus - q_minus) / (2.0 * epsilon)
    return grad


def finite_diff_hessian_vv(
    u: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
    epsilon: float = 0.005,
    use_gaussian: bool = False,
) -> float:
    """Estimate directional second derivative D²q[v,v] by finite differences.

    Parameters
    ----------
    u : Portfolio weight vector.
    v : Direction vector.
    alpha : Quantile level.
    mu, A : Model parameters.
    epsilon : Step size.
    use_gaussian : If True compute Gaussian analytic quantile.

    Returns
    -------
    float: D²q[v,v] estimate.
    """
    def q(w):
        if use_gaussian:
            return gaussian_quantile_analytic(w, alpha, mu, A)
        return quantile_lognormal_deterministic(w, alpha, mu, A)

    return (q(u + epsilon * v) - 2.0 * q(u) + q(u - epsilon * v)) / epsilon**2


def hessian_vv_formula(
    u: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: np.ndarray,
    A: np.ndarray,
    delta: float = 0.005,
) -> float:
    """Compute D²q[v,v] via the conditional variance formula.

    D²q[v,v] = -f_Y(y)^{-1} * d/dy[ f_Y(y) Var(v^T X | u^T X = y) ] @ y=q

    Parameters
    ----------
    u : Portfolio weight.
    v : Direction.
    alpha : Quantile level.
    mu, A : Model parameters.
    delta : Step for y-derivative.

    Returns
    -------
    float: Formula estimate.
    """
    y0 = quantile_lognormal_deterministic(u, alpha, mu, A)

    def F(y):
        fY = pdf_Y_lognormal(y, u, mu, A)
        var_vX = conditional_variance_surface(y, u, v, mu, A)
        return fY * var_vX

    f_y0 = pdf_Y_lognormal(y0, u, mu, A)
    d_F = (F(y0 + delta) - F(y0 - delta)) / (2.0 * delta)
    if abs(f_y0) < 1e-30:
        return np.nan
    return -d_F / f_y0


def run_gaussian_validation(
    A: np.ndarray,
    mu: np.ndarray,
    u_test_points: list,
    alphas: list,
) -> pd.DataFrame:
    """Validate gradient and Hessian for Gaussian model (exact analytic available).

    Parameters
    ----------
    A : Covariance matrix.
    mu : Mean.
    u_test_points : List of u = (t, 1-t) tuples.
    alphas : Alpha levels.

    Returns
    -------
    pd.DataFrame with errors.
    """
    rows = []
    for u_tuple in u_test_points:
        u = np.array(u_tuple)
        v = np.array([1.0, -1.0])  # direction along simplex
        v = v / np.linalg.norm(v)
        for alpha in alphas:
            # Analytic
            q_ana = gaussian_quantile_analytic(u, alpha, mu, A)
            grad_ana = gaussian_quantile_gradient_analytic(u, alpha, mu, A)
            hess_ana = gaussian_quantile_hessian_vv_analytic(u, v, alpha, mu, A)

            # Finite differences
            grad_fd = finite_diff_gradient(u, alpha, mu, A, epsilon=0.002, use_gaussian=True)
            hess_fd = finite_diff_hessian_vv(u, v, alpha, mu, A, epsilon=0.005, use_gaussian=True)

            grad_err = float(np.linalg.norm(grad_fd - grad_ana))
            hess_err = abs(hess_fd - hess_ana)

            # Conditional expectation formula gradient
            # For Gaussian Y ~ N(u^T mu, u^T A u), q = mu_Y + z_alpha sigma_Y
            # Gradient = mu + z_alpha * (A u) / sigma_Y
            grad_cond = grad_ana  # same for Gaussian
            cond_err = float(np.linalg.norm(grad_fd - grad_cond))

            rows.append({
                'u': u.tolist(),
                'alpha': alpha,
                'q_analytic': round(q_ana, 6),
                'grad_err_fd_vs_analytic': round(grad_err, 8),
                'hess_fd': round(hess_fd, 6),
                'hess_analytic': round(hess_ana, 6),
                'hess_err': round(hess_err, 6),
                'model': 'Gaussian',
            })
    return pd.DataFrame(rows)


def run_lognormal_validation(
    sc_name: str,
    u_test_points: list,
    alphas: list,
) -> pd.DataFrame:
    """Validate gradient and Hessian for lognormal model numerically.

    Parameters
    ----------
    sc_name : Scenario name.
    u_test_points : Test portfolio points.
    alphas : Alpha levels.

    Returns
    -------
    pd.DataFrame with errors.
    """
    sc = setup_scenario(sc_name)
    mu, A = sc['mu'], sc['A']
    rows = []

    for u_tuple in u_test_points:
        u = np.array(u_tuple)
        v = np.array([1.0, -1.0])
        v = v / np.linalg.norm(v)
        for alpha in alphas:
            q_det = quantile_lognormal_deterministic(u, alpha, mu, A)

            # Gradient by FD
            grad_fd = finite_diff_gradient(u, alpha, mu, A, epsilon=0.005, use_gaussian=False)

            # Gradient via conditional expectation formula (surface integration)
            grad_cond = conditional_expectation_surface(q_det, u, mu, A)

            grad_err = float(np.linalg.norm(grad_fd - grad_cond))

            # Hessian FD
            hess_fd = finite_diff_hessian_vv(u, v, alpha, mu, A, epsilon=0.005, use_gaussian=False)

            # Hessian via formula
            hess_formula = hessian_vv_formula(u, v, alpha, mu, A, delta=0.01)

            hess_err = abs(hess_fd - hess_formula) if not np.isnan(hess_formula) else np.nan

            rows.append({
                'scenario': sc_name,
                'u': u.tolist(),
                'alpha': alpha,
                'q_det': round(q_det, 6),
                'grad_fd': [round(g, 6) for g in grad_fd],
                'grad_cond': [round(g, 6) for g in grad_cond],
                'grad_err': round(grad_err, 6),
                'hess_fd': round(hess_fd, 6),
                'hess_formula': round(hess_formula, 6) if not np.isnan(hess_formula) else np.nan,
                'hess_err': round(hess_err, 6) if not np.isnan(hess_err) else np.nan,
                'model': 'Lognormal',
            })
    return pd.DataFrame(rows)


def run_exp9() -> dict:
    """Run Experiment 9: Gradient and Hessian validation."""
    print("=" * 60)
    print("Experiment 9: Quantile Derivative and Hessian Validation")
    print("=" * 60)

    # Gaussian benchmark
    sc_s1 = setup_scenario('S1')
    A_gauss = sc_s1['A']
    mu_gauss = sc_s1['mu']

    u_test = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    alphas_test = [0.1, 0.5, 0.9]

    print("\n  --- Gaussian validation ---")
    df_gauss = run_gaussian_validation(A_gauss, mu_gauss, u_test, alphas_test)
    print(df_gauss[['u', 'alpha', 'q_analytic', 'grad_err_fd_vs_analytic',
                      'hess_fd', 'hess_analytic', 'hess_err']].to_string(index=False))

    print("\n  --- Lognormal validation (S1) ---")
    df_lognorm = run_lognormal_validation('S1', u_test, alphas_test)
    print(df_lognorm[['scenario', 'u', 'alpha', 'q_det', 'grad_err',
                        'hess_fd', 'hess_formula', 'hess_err']].to_string(index=False))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(range(len(df_gauss)), df_gauss['grad_err_fd_vs_analytic'], color='steelblue')
    ax.set_ylabel('Gradient error (FD vs analytic)')
    ax.set_title('Gaussian: Gradient validation')
    ax.set_xticks(range(len(df_gauss)))
    ax.set_xticklabels([f"α={row.alpha}" for _, row in df_gauss.iterrows()], rotation=45, ha='right', fontsize=7)

    ax = axes[1]
    valid_lognorm = df_lognorm.dropna(subset=['hess_err'])
    ax.bar(range(len(valid_lognorm)), valid_lognorm['hess_err'].abs(), color='firebrick')
    ax.set_ylabel('Hessian error |FD - formula|')
    ax.set_title('Lognormal: Hessian validation')
    ax.set_xticks(range(len(valid_lognorm)))
    ax.set_xticklabels([f"α={row.alpha}" for _, row in valid_lognorm.iterrows()], rotation=45, ha='right', fontsize=7)

    plt.suptitle('Experiment 9: Derivative and Hessian Validation')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp9_derivative_validation.png', dpi=120)
    plt.close(fig)

    # Assertions (tight tolerances reflecting actual numerical accuracy)
    assert (df_gauss['grad_err_fd_vs_analytic'] < 1e-4).all(), \
        f"Gaussian gradient FD errors too large: {df_gauss['grad_err_fd_vs_analytic'].max()}"
    assert (df_gauss['hess_err'] < 1e-3).all(), \
        f"Gaussian Hessian errors too large: {df_gauss['hess_err'].max()}"
    assert (df_lognorm['grad_err'] < 1e-3).all(), \
        f"Lognormal gradient errors too large: {df_lognorm['grad_err'].max()}"
    valid_hess = df_lognorm.dropna(subset=['hess_err'])
    assert (valid_hess['hess_err'] < 5e-3).all(), \
        f"Lognormal Hessian errors too large: {valid_hess['hess_err'].max()}"

    print("\n✓ Experiment 9 passed: gradient and Hessian formulas validated.")
    return {'df_gauss': df_gauss, 'df_lognorm': df_lognorm}


if __name__ == '__main__':
    run_exp9()
