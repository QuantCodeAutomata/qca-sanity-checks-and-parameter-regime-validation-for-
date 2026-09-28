"""
Experiment 4: Lognormal left-tail corner regime (S3) — concavity and convergence
to the linear corner shape q_0(t) = t, with optimizer → (1, 0).

Replicates paper Figure 3 left panel.
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
from src.quantile import empirical_quantile_curve, compute_curvature
from src.utils import (
    simplex_grid, normalized_shape, find_optimizer, shape_error_metrics
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

ALPHAS = [0.1, 0.01, 1e-5]
N_MC = 2_000_000
N_GRID = 201


def run_exp4() -> dict:
    """Run Experiment 4: Left-tail corner regime for scenario S3."""
    print("=" * 60)
    print("Experiment 4: Lognormal Left-Tail Corner Regime (S3)")
    print("=" * 60)

    sc = setup_scenario('S3')
    mu, A = sc['mu'], sc['A']
    b, beta = sc['b'], sc['beta']
    print(f"  S3: b={b}, beta={beta:.6f}, regime={sc['regime']}")
    print(f"  Theoretical left-tail optimizer: t* = 1.0 (corner u=(1,0))")

    t_grid = simplex_grid(N_GRID)
    dt = t_grid[1] - t_grid[0]
    t_int = t_grid[1:-1]

    X = generate_lognormal_samples(mu, A, N_MC, seed=42)

    # Theoretical asymptotic shape q_0(t) = t
    q_theory = t_grid.copy()

    q_curves = {}
    q_norm_curves = {}
    curv_curves = {}
    k_alphas = {}
    optimizers = {}
    shape_errors = {}

    for alpha in ALPHAS:
        q = empirical_quantile_curve(X, t_grid, alpha)
        # Normalize by k_alpha = q_alpha(1, 0) ≈ corner value
        k = float(q[-1]) if q[-1] > 0 else float(np.max(q))
        q_norm = q / k if k > 0 else q
        k_alphas[alpha] = k
        q_curves[alpha] = q
        q_norm_curves[alpha] = q_norm
        curv_curves[alpha] = compute_curvature(q, dt)
        optimizers[alpha] = find_optimizer(q, t_grid)
        # Shape error vs q_0(t) = t, evaluated on interior
        shape_errors[alpha] = shape_error_metrics(q_norm, q_theory, t_grid, eps=0.02)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(ALPHAS)))

    ax = axes[0, 0]
    for i, alpha in enumerate(ALPHAS):
        ax.plot(t_grid, q_curves[alpha], color=colors[i], label=f'α={alpha}')
    ax.set_xlabel('t')
    ax.set_ylabel('q_α(t)')
    ax.set_title('S3: Left-tail quantile curves')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t_grid, t_grid, 'k--', linewidth=2, label='Theory q_0(t)=t')
    for i, alpha in enumerate(ALPHAS):
        ax.plot(t_grid, q_norm_curves[alpha], color=colors[i], label=f'α={alpha}')
    ax.set_xlabel('t')
    ax.set_ylabel('q_α(t)/k_α')
    ax.set_title('S3: Normalized left-tail shapes vs q_0(t)=t')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, alpha in enumerate(ALPHAS):
        ax.plot(t_int, curv_curves[alpha], color=colors[i], label=f'α={alpha}')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel('d²q/dt²')
    ax.set_title('S3: Left-tail curvature (negative = concave)')
    ax.legend()

    ax = axes[1, 1]
    t_star_list = [optimizers[a] for a in ALPHAS]
    ax.semilogx(ALPHAS, t_star_list, 'o-', color='royalblue', label='Empirical t*_α')
    ax.axhline(1.0, color='r', linestyle='--', label='Theory t* = 1.0')
    ax.set_xlabel('α')
    ax.set_ylabel('t*_α')
    ax.set_title('S3: Optimizer convergence (left tail)')
    ax.legend()

    plt.suptitle('Experiment 4: Lognormal Left-Tail Corner Regime (S3)')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp4_corner_left_S3.png', dpi=120)
    plt.close(fig)

    # Summary
    interior = slice(1, -1)
    summary_rows = []
    for alpha in ALPHAS:
        t_star_emp = optimizers[alpha]
        se = shape_errors[alpha]
        frac_neg = float(np.mean(curv_curves[alpha] < 0))
        # Chord-based concavity: all interior q-values lie at/above the chord.
        q_curve = q_curves[alpha]
        chord = q_curve[0] + (q_curve[-1] - q_curve[0]) * t_grid[interior]
        chord_concave = bool(np.all(q_curve[interior] >= chord - 1e-6 * np.abs(chord)))
        summary_rows.append({
            'alpha': alpha,
            't_star_emp': round(t_star_emp, 6),
            't_star_theory': 1.0,
            'opt_err': round(abs(t_star_emp - 1.0), 6),
            'shape_sup_norm': round(se['sup_norm'], 6),
            'shape_sup_rel': round(se['sup_rel'], 6),
            'frac_neg_curv': round(frac_neg, 4),
            'chord_concave': chord_concave,
        })

    df = pd.DataFrame(summary_rows)
    print("\nSummary table:")
    print(df.to_string(index=False))

    for _, row in df.iterrows():
        assert row['chord_concave'], \
            f"Expected chord-concavity in S3 left tail at alpha={row['alpha']}"
        if row['alpha'] <= 0.01:
            assert row['opt_err'] < 0.3, \
                f"Optimizer not converging to t=1: err={row['opt_err']}"

    print("\n✓ Experiment 4 passed: left-tail corner concavity confirmed.")
    return {'summary': df, 'q_curves': q_curves, 'q_norm_curves': q_norm_curves,
            't_grid': t_grid, 'shape_errors': shape_errors}


if __name__ == '__main__':
    run_exp4()
