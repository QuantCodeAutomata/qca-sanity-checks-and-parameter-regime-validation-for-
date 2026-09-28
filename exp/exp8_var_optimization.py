"""
Experiment 8: Tail-region VaR optimization on the simplex and verification
of asymptotic optimizer formulas.

Compares empirical argmax of q_alpha(u) to theoretical predictions from
b/beta and Section 5 A-matrix formulas.
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
from src.utils import simplex_grid, find_optimizer

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_MC = 5_000_000
N_GRID = 401


def section5_optimizer_formula(A: np.ndarray) -> dict:
    """Compute Section 5 two-asset optimizer formula in terms of A.

    For the 2D barycentric case:
        t* = (A_22 - A_12) / (A_11 + A_22 - 2*A_12)

    Parameters
    ----------
    A : 2x2 SPD covariance matrix.

    Returns
    -------
    dict with 't_star_sec5' and 'u_star_sec5'.
    """
    A11, A12, A22 = A[0, 0], A[0, 1], A[1, 1]
    denom = A11 + A22 - 2.0 * A12
    t_star = (A22 - A12) / denom if abs(denom) > 1e-14 else 0.5
    return {'t_star_sec5': float(t_star), 'u_star_sec5': np.array([t_star, 1.0 - t_star])}


def run_optimizer_analysis(
    scenario_name: str,
    alphas: list,
    N: int = N_MC,
    seed: int = 42,
) -> dict:
    """Compute empirical optimizers vs theoretical targets for all alphas.

    Parameters
    ----------
    scenario_name : str
        'S1', 'S2', or 'S3'.
    alphas : List of alpha levels.
    N : Monte Carlo draws.
    seed : Random seed.

    Returns
    -------
    dict with per-alpha optimizer info.
    """
    sc = setup_scenario(scenario_name)
    mu, A, B, b, beta = sc['mu'], sc['A'], sc['B'], sc['b'], sc['beta']
    regime = sc['regime']
    t_star_theory = sc['optimizer']['t_star']

    t_grid = simplex_grid(N_GRID)
    X = generate_lognormal_samples(mu, A, N, seed=seed)

    optimizers = {}
    q_curves = {}
    for alpha in alphas:
        q = empirical_quantile_curve(X, t_grid, alpha)
        t_star_emp = find_optimizer(q, t_grid)
        q_curves[alpha] = q
        optimizers[alpha] = {
            't_star_emp': t_star_emp,
            'opt_err': abs(t_star_emp - t_star_theory),
        }

    # Section 5 formula (barycentric scenarios only)
    sec5 = section5_optimizer_formula(A) if regime == 'barycentric' else None

    return {
        'scenario': scenario_name,
        'sc': sc,
        't_star_theory': t_star_theory,
        'regime': regime,
        't_grid': t_grid,
        'q_curves': q_curves,
        'optimizers': optimizers,
        'sec5': sec5,
        'b': b,
        'beta': beta,
    }


def plot_optimizer_results(res: dict, output_dir: Path) -> None:
    """Produce quantile-curve and optimizer-path plots."""
    name = res['scenario']
    t_grid = res['t_grid']
    alphas = sorted(res['q_curves'].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for i, alpha in enumerate(alphas):
        q = res['q_curves'][alpha]
        ax.plot(t_grid, q, color=colors[i], label=f'α={alpha}')
    ax.axvline(res['t_star_theory'], color='r', linestyle='--', linewidth=1.5,
               label=f'Theory t*={res["t_star_theory"]:.4f}')
    ax.set_xlabel('t')
    ax.set_ylabel('q_α(t)')
    ax.set_title(f'{name}: Quantile curves and optimizer')
    ax.legend(fontsize=8)

    ax = axes[1]
    t_stars = [res['optimizers'][a]['t_star_emp'] for a in alphas]
    ax.plot([str(a) for a in alphas], t_stars, 'o-', color='steelblue', label='Empirical t*_α')
    ax.axhline(res['t_star_theory'], color='r', linestyle='--', linewidth=1.5,
               label=f'Theory t*={res["t_star_theory"]:.4f}')
    if res['sec5'] is not None:
        ax.axhline(res['sec5']['t_star_sec5'], color='green', linestyle=':', linewidth=1.5,
                   label=f'Sec5 t*={res["sec5"]["t_star_sec5"]:.4f}')
    ax.set_xlabel('α')
    ax.set_ylabel('t*_α')
    ax.set_title(f'{name}: Optimizer convergence')
    ax.legend(fontsize=8)

    plt.suptitle(f'Experiment 8: VaR Optimizer — {name}')
    plt.tight_layout()
    fig.savefig(output_dir / f'exp8_optimizer_{name}.png', dpi=120)
    plt.close(fig)


def run_exp8() -> dict:
    """Run Experiment 8: VaR optimization for S1, S2, S3."""
    print("=" * 60)
    print("Experiment 8: VaR Optimization on Simplex")
    print("=" * 60)

    all_results = {}
    summary_rows = []

    scenario_alphas = {
        'S1': [0.1, 0.01, 1e-5],
        'S2': [0.1, 0.01, 1e-5],
        'S3_left': [0.1, 0.01, 1e-5],
        'S3_right': [0.9, 0.99, 0.9999],
    }

    # Right-tail corner theory target: t*=0 (u=(0,1)) for S3_right
    RIGHT_TAIL_THEORY = {'S3_right': 0.0}

    for tag, alphas in scenario_alphas.items():
        base_name = tag.split('_')[0]
        print(f"\n  Running {tag} ...")
        res = run_optimizer_analysis(base_name, alphas, N=N_MC)
        # Override theory target for right-tail corner case
        if tag in RIGHT_TAIL_THEORY:
            t_theory = RIGHT_TAIL_THEORY[tag]
            # Recompute opt_err against the right-tail target
            for alpha in alphas:
                res['optimizers'][alpha]['opt_err'] = abs(
                    res['optimizers'][alpha]['t_star_emp'] - t_theory
                )
        else:
            t_theory = res['t_star_theory']
        plot_optimizer_results(res, RESULTS_DIR)
        all_results[tag] = res

        for alpha in alphas:
            opt_info = res['optimizers'][alpha]
            row = {
                'tag': tag,
                'alpha': alpha,
                't_star_emp': round(opt_info['t_star_emp'], 6),
                't_star_theory': round(t_theory, 6),
                'opt_err': round(opt_info['opt_err'], 6),
                'regime': res['regime'],
            }
            # Check Section 5 / b/beta equivalence for barycentric
            if res['sec5'] is not None:
                t_b_beta = res['sc']['b'][0] / res['sc']['beta']
                t_sec5 = res['sec5']['t_star_sec5']
                row['t_star_b_beta'] = round(t_b_beta, 6)
                row['t_star_sec5'] = round(t_sec5, 6)
                row['sec5_equiv_err'] = round(abs(t_b_beta - t_sec5), 8)
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    print("\nSummary table:")
    cols = [c for c in ['tag', 'alpha', 't_star_emp', 't_star_theory', 'opt_err',
                         't_star_b_beta', 't_star_sec5', 'sec5_equiv_err'] if c in df.columns]
    print(df[cols].to_string(index=False))

    # Assertions
    for _, row in df.iterrows():
        # Check left-tail and right-tail convergence at moderate-to-deep tail
        is_right_tail = str(row['tag']).endswith('_right')
        if is_right_tail:
            # Right-tail: assert optimizer near t=0 for alpha >= 0.9
            if row['alpha'] >= 0.9:
                assert row['opt_err'] < 0.15, \
                    f"Right-tail optimizer not converging to t=0 at {row['tag']} " \
                    f"alpha={row['alpha']}: t*={row['t_star_emp']:.4f}"
        else:
            # Left-tail: optimizer should converge to theory at deep tail
            if row['alpha'] <= 0.01:
                assert row['opt_err'] < 0.35, \
                    f"Optimizer too far from theory at {row['tag']} alpha={row['alpha']}: " \
                    f"{row['opt_err']:.4f}"
        if 'sec5_equiv_err' in row.index and pd.notna(row['sec5_equiv_err']):
            assert row['sec5_equiv_err'] < 1e-6, \
                f"Section 5 and b/beta formulas don't match: {row['sec5_equiv_err']}"

    print("\n✓ Experiment 8 passed: optimizers converge to theory; Section 5 ≡ b/beta.")
    return {'summary': df, 'all_results': all_results}


if __name__ == '__main__':
    run_exp8()
