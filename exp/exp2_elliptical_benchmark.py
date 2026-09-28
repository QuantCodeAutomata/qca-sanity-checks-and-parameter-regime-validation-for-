"""
Experiment 2: Elliptical benchmark — quantile concavity in the left tail and
convexity in the right tail for linear portfolios under normal distribution.
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
from src.quantile import (
    empirical_quantile_curve, gaussian_quantile_on_simplex, compute_curvature
)
from src.utils import simplex_grid

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_gaussian_curvature_analysis(
    A: np.ndarray,
    mu: np.ndarray,
    alphas_low: list,
    alphas_high: list,
    n_grid: int = 201,
    N_mc: int = 1_000_000,
    seed: int = 42
) -> dict:
    """Compute analytic and empirical quantile curvature for a Gaussian scenario.

    Parameters
    ----------
    A : Covariance matrix.
    mu : Mean vector.
    alphas_low : Left-tail alpha values.
    alphas_high : Right-tail alpha values.
    n_grid : Grid points on simplex.
    N_mc : Monte Carlo sample size.
    seed : Random seed.

    Returns
    -------
    dict with curves and curvature arrays.
    """
    t_grid = simplex_grid(n_grid)
    dt = t_grid[1] - t_grid[0]

    # Monte Carlo sample (Gaussian, not lognormal here)
    rng = np.random.default_rng(seed)
    X_mc = rng.multivariate_normal(mu, A, size=N_mc)

    results = {}
    all_alphas = alphas_low + alphas_high
    for alpha in all_alphas:
        q_analytic = gaussian_quantile_on_simplex(A, mu, alpha, t_grid)
        q_mc = empirical_quantile_curve(X_mc, t_grid, alpha)
        curv_analytic = compute_curvature(q_analytic, dt)
        curv_mc = compute_curvature(q_mc, dt)
        results[alpha] = {
            'q_analytic': q_analytic,
            'q_mc': q_mc,
            'curv_analytic': curv_analytic,
            'curv_mc': curv_mc,
            'frac_neg_analytic': float(np.mean(curv_analytic < 0)),
            'frac_pos_analytic': float(np.mean(curv_analytic > 0)),
            'frac_neg_mc': float(np.mean(curv_mc < 0)),
            'frac_pos_mc': float(np.mean(curv_mc > 0)),
        }
    return {'t_grid': t_grid, 'dt': dt, 'curves': results}


def plot_gaussian_quantiles(res: dict, alphas_low: list, alphas_high: list,
                             scenario_name: str, output_dir: Path) -> None:
    """Plot quantile curves and curvature for the Gaussian benchmark."""
    t_grid = res['t_grid']
    curves = res['curves']
    t_int = t_grid[1:-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Left-tail quantile curves ---
    ax = axes[0, 0]
    for alpha in alphas_low:
        ax.plot(t_grid, curves[alpha]['q_analytic'], label=f'α={alpha} (analytic)')
        ax.plot(t_grid, curves[alpha]['q_mc'], '--', alpha=0.6, label=f'α={alpha} (MC)')
    ax.set_xlabel('t')
    ax.set_ylabel('q_α(t, 1-t)')
    ax.set_title(f'Left-tail quantiles — {scenario_name}')
    ax.legend(fontsize=8)

    # --- Right-tail quantile curves ---
    ax = axes[0, 1]
    for alpha in alphas_high:
        ax.plot(t_grid, curves[alpha]['q_analytic'], label=f'α={alpha} (analytic)')
        ax.plot(t_grid, curves[alpha]['q_mc'], '--', alpha=0.6, label=f'α={alpha} (MC)')
    ax.set_xlabel('t')
    ax.set_ylabel('q_α(t, 1-t)')
    ax.set_title(f'Right-tail quantiles — {scenario_name}')
    ax.legend(fontsize=8)

    # --- Left-tail curvature (analytic) ---
    ax = axes[1, 0]
    for alpha in alphas_low:
        ax.plot(t_int, curves[alpha]['curv_analytic'], label=f'α={alpha}')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel("d²q/dt²")
    ax.set_title(f'Left-tail curvature (analytic) — {scenario_name}')
    ax.legend(fontsize=8)

    # --- Right-tail curvature (analytic) ---
    ax = axes[1, 1]
    for alpha in alphas_high:
        ax.plot(t_int, curves[alpha]['curv_analytic'], label=f'α={alpha}')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel("d²q/dt²")
    ax.set_title(f'Right-tail curvature (analytic) — {scenario_name}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'exp2_gaussian_benchmark_{scenario_name}.png', dpi=120)
    plt.close(fig)


def run_exp2() -> dict:
    """Run Experiment 2: Gaussian elliptical benchmark curvature analysis."""
    print("=" * 60)
    print("Experiment 2: Elliptical Benchmark — Curvature Analysis")
    print("=" * 60)

    alphas_low = [0.1, 0.01]
    alphas_high = [0.9, 0.99]

    # Use S1 covariance (A = B^{-1} for S1)
    sc1 = setup_scenario('S1')
    A_s1 = sc1['A']
    mu_s1 = sc1['mu']

    # Also test with a simple A = [[1, 0.5], [0.5, 1]]
    A_simple = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu_zero = np.array([0.0, 0.0])

    scenarios_to_test = [
        ('S1_cov', A_s1, mu_s1),
        ('simple', A_simple, mu_zero),
    ]

    summary_rows = []
    all_results = {}

    for sc_name, A, mu in scenarios_to_test:
        print(f"\n  Scenario: {sc_name}")
        res = run_gaussian_curvature_analysis(
            A, mu, alphas_low, alphas_high, n_grid=201, N_mc=500_000
        )
        plot_gaussian_quantiles(res, alphas_low, alphas_high, sc_name, RESULTS_DIR)
        all_results[sc_name] = res

        for alpha in alphas_low:
            c = res['curves'][alpha]
            majority_concave = c['frac_neg_analytic'] > 0.8
            summary_rows.append({
                'scenario': sc_name,
                'alpha': alpha,
                'tail': 'left',
                'frac_neg_curv': c['frac_neg_analytic'],
                'majority_concave': majority_concave,
            })
            print(f"    α={alpha} (left tail): frac concave = {c['frac_neg_analytic']:.3f}")

        for alpha in alphas_high:
            c = res['curves'][alpha]
            majority_convex = c['frac_pos_analytic'] > 0.8
            summary_rows.append({
                'scenario': sc_name,
                'alpha': alpha,
                'tail': 'right',
                'frac_pos_curv': c['frac_pos_analytic'],
                'majority_convex': majority_convex,
            })
            print(f"    α={alpha} (right tail): frac convex = {c['frac_pos_analytic']:.3f}")

    df_summary = pd.DataFrame(summary_rows)
    print("\nCurvature summary table:")
    print(df_summary.to_string(index=False))

    # Validate: left tail → concave, right tail → convex
    for _, row in df_summary.iterrows():
        if row['tail'] == 'left':
            assert row['frac_neg_curv'] > 0.8, \
                f"Left tail should be mostly concave: {row}"
        else:
            assert row['frac_pos_curv'] > 0.8, \
                f"Right tail should be mostly convex: {row}"

    print("\n✓ All Experiment 2 checks passed (left-tail concave, right-tail convex).")
    return {'summary': df_summary, 'all_results': all_results}


if __name__ == '__main__':
    run_exp2()
