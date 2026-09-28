"""
Experiment 7: Tail universality and location-parameter invariance of normalized
quantile shapes.

Tests the paper's scale-shape separation: q_alpha(u) ~ k_alpha * q_0(u), where
the shape q_0 depends only on A (not on mu).
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
from src.quantile import empirical_quantile_curve
from src.regime import theoretical_normalized_shape_barycentric
from src.utils import simplex_grid, normalized_shape, shape_error_metrics

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

MU_VECTORS = [
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.5]),
    np.array([1.0, 1.0]),
]
MU_LABELS = ['(0,0)', '(1,0)', '(0,1)', '(-1,0.5)', '(1,1)']

ALPHAS_LEFT = [0.1, 0.01, 1e-5]
ALPHAS_RIGHT = [0.9, 0.99, 0.9999]
N_MC = 2_000_000
N_GRID = 201


def compute_normalized_curves_for_mu(
    A: np.ndarray,
    mu: np.ndarray,
    alphas: list,
    t_grid: np.ndarray,
    N: int,
    seed: int = 42,
    normalize_corner: bool = False,
) -> dict:
    """Compute normalized quantile curves for given mu and A.

    Parameters
    ----------
    A : Covariance matrix.
    mu : Mean vector.
    alphas : List of alpha levels.
    t_grid : Simplex grid.
    N : Monte Carlo sample size.
    seed : Random seed.
    normalize_corner : If True, normalize by corner value, else by max.

    Returns
    -------
    dict {alpha: {'raw': ..., 'norm': ..., 'k': ...}}
    """
    X = generate_lognormal_samples(mu, A, N, seed=seed)
    results = {}
    for alpha in alphas:
        q = empirical_quantile_curve(X, t_grid, alpha)
        if normalize_corner:
            # For right-tail corner: normalize by q at t=0
            k = float(q[0]) if q[0] > 0 else float(np.max(q))
        else:
            k = float(np.max(q))
        q_norm = q / k if k > 0 else q
        results[alpha] = {'raw': q, 'norm': q_norm, 'k': k}
    return results


def cross_mu_collapse_metric(
    norm_curves_by_mu: dict,
    alpha: float,
    t_grid: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """Compute max cross-mu sup-norm of normalized shapes.

    Parameters
    ----------
    norm_curves_by_mu : {label: curves_dict}.
    alpha : Alpha level.
    t_grid : Grid.
    eps : Interior margin.

    Returns
    -------
    float: max_{mu_a, mu_b} sup_t |q_norm^{mu_a}(t) - q_norm^{mu_b}(t)|
    """
    mask = (t_grid >= eps) & (t_grid <= 1 - eps)
    labels = list(norm_curves_by_mu.keys())
    max_diff = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            q_a = norm_curves_by_mu[labels[i]][alpha]['norm'][mask]
            q_b = norm_curves_by_mu[labels[j]][alpha]['norm'][mask]
            diff = float(np.max(np.abs(q_a - q_b)))
            max_diff = max(max_diff, diff)
    return max_diff


def run_exp7() -> dict:
    """Run Experiment 7: Tail universality and mu-invariance."""
    print("=" * 60)
    print("Experiment 7: Tail Universality and Location-Parameter Invariance")
    print("=" * 60)

    sc = setup_scenario('S1')
    A = sc['A']
    b, beta = sc['b'], sc['beta']
    t_grid = simplex_grid(N_GRID)

    # Theoretical shape for S1 barycentric
    q_theory = theoretical_normalized_shape_barycentric(t_grid, b, beta)
    k_theory = float(np.max(q_theory))
    q_theory_norm = q_theory / k_theory

    print(f"\n  Fixed A = S1 covariance, b={b}, beta={beta:.6f}")
    print(f"  Testing {len(MU_VECTORS)} mu vectors: {MU_LABELS}")

    # Compute normalized curves for each mu
    all_mu_curves = {}
    for label, mu in zip(MU_LABELS, MU_VECTORS):
        print(f"    mu={label} ... ", end='', flush=True)
        curves = compute_normalized_curves_for_mu(A, mu, ALPHAS_LEFT, t_grid, N_MC)
        all_mu_curves[label] = curves
        print("done")

    # Compute cross-mu collapse metrics
    summary_rows = []
    for alpha in ALPHAS_LEFT:
        cross_mu_err = cross_mu_collapse_metric(all_mu_curves, alpha, t_grid)
        # Shape error vs theory for mu=(0,0)
        q_norm_ref = all_mu_curves['(0,0)'][alpha]['norm']
        se = shape_error_metrics(q_norm_ref, q_theory_norm, t_grid)
        summary_rows.append({
            'alpha': alpha,
            'cross_mu_sup_norm': round(cross_mu_err, 6),
            'shape_vs_theory_sup_norm': round(se['sup_norm'], 6),
            'shape_vs_theory_sup_rel': round(se['sup_rel'], 6),
            'shapes_collapse': cross_mu_err < 0.5,
        })

    df = pd.DataFrame(summary_rows)
    print("\nSummary table:")
    print(df.to_string(index=False))

    # --- Plots ---
    fig, axes = plt.subplots(len(ALPHAS_LEFT), 2, figsize=(14, 5 * len(ALPHAS_LEFT)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MU_LABELS)))

    for row_i, alpha in enumerate(ALPHAS_LEFT):
        # Raw quantile curves
        ax = axes[row_i, 0]
        for j, (label, mu) in enumerate(zip(MU_LABELS, MU_VECTORS)):
            q_raw = all_mu_curves[label][alpha]['raw']
            ax.plot(t_grid, q_raw, color=colors[j], label=f'mu={label}')
        ax.set_xlabel('t')
        ax.set_ylabel('q_α(t)')
        ax.set_title(f'α={alpha}: Raw quantile curves (vary mu)')
        ax.legend(fontsize=7)

        # Normalized shapes
        ax = axes[row_i, 1]
        ax.plot(t_grid, q_theory_norm, 'k--', linewidth=2, label='Theory q_0')
        for j, (label, mu) in enumerate(zip(MU_LABELS, MU_VECTORS)):
            q_norm = all_mu_curves[label][alpha]['norm']
            ax.plot(t_grid, q_norm, color=colors[j], label=f'mu={label}', alpha=0.8)
        ax.set_xlabel('t')
        ax.set_ylabel('q_α(t)/k_α')
        ax.set_title(f'α={alpha}: Normalized shapes (collapse expected)')
        ax.legend(fontsize=7)

    plt.suptitle('Experiment 7: Tail Universality — mu Invariance (S1 A fixed)')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp7_tail_universality_S1.png', dpi=120)
    plt.close(fig)

    # Assertions
    for _, row in df.iterrows():
        if row['alpha'] <= 0.01:
            assert row['shapes_collapse'], \
                f"Shapes not collapsing at alpha={row['alpha']}: cross_mu_err={row['cross_mu_sup_norm']}"

    print("\n✓ Experiment 7 passed: normalized shapes collapse across mu values.")
    return {'summary': df, 'all_mu_curves': all_mu_curves, 't_grid': t_grid}


if __name__ == '__main__':
    run_exp7()
