"""
Experiment 1: Sanity checks and parameter-regime validation for synthetic lognormal
quantile-surface experiments.

Validates: SPD checks, matrix inversion consistency, b/beta computation,
regime classification, positive homogeneity of quantiles, and dominance
equivalence in 2D.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    SCENARIOS, validate_spd, compute_A_B, compute_b_beta,
    classify_regime, theoretical_optimizer, generate_lognormal_samples,
    setup_scenario,
)
from src.quantile import empirical_quantile_left
from src.regime import verify_dominance_equivalence
from src.utils import check_homogeneity

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_matrix_inversion_checks() -> pd.DataFrame:
    """Check A = B^{-1} and B = A^{-1} to numerical tolerance for all scenarios."""
    rows = []
    for name in ['S1', 'S2', 'S3']:
        sc = setup_scenario(name)
        A, B = sc['A'], sc['B']
        I = np.eye(2)
        err_AB = float(np.linalg.norm(A @ B - I, 'fro'))
        err_BA = float(np.linalg.norm(B @ A - I, 'fro'))
        rows.append({
            'scenario': name,
            '||AB - I||_F': err_AB,
            '||BA - I||_F': err_BA,
            'near_machine_eps': err_AB < 1e-13 and err_BA < 1e-13,
        })
    return pd.DataFrame(rows)


def run_b_beta_checks() -> pd.DataFrame:
    """Compute b and beta for all scenarios and compare to paper values."""
    expected = {
        'S1': {'b': np.array([0.6, 0.35]), 'beta': 0.95, 't_star': 0.6315789473684210},
        'S2': {'b': np.array([5.1, 0.35]), 'beta': 5.45, 't_star': 0.9357798165137615},
        'S3': {'b': np.array([0.2, -0.05]), 'beta': 0.15},
    }
    rows = []
    for name in ['S1', 'S2', 'S3']:
        sc = setup_scenario(name)
        b, beta = sc['b'], sc['beta']
        exp = expected[name]
        b_err = float(np.max(np.abs(b - exp['b'])))
        beta_err = abs(beta - exp['beta'])
        row = {
            'scenario': name,
            'b_computed': b.tolist(),
            'b_expected': exp['b'].tolist(),
            'b_max_err': b_err,
            'beta_computed': round(beta, 12),
            'beta_expected': exp['beta'],
            'beta_err': beta_err,
            'b_ok': b_err < 1e-10,
            'beta_ok': beta_err < 1e-10,
        }
        if 't_star' in exp:
            t_star = sc['optimizer']['t_star']
            row['t_star_computed'] = round(t_star, 15)
            row['t_star_expected'] = exp['t_star']
            row['t_star_err'] = abs(t_star - exp['t_star'])
        rows.append(row)
    return pd.DataFrame(rows)


def run_regime_classification() -> pd.DataFrame:
    """Verify regime classification for all scenarios."""
    expected_regimes = {'S1': 'barycentric', 'S2': 'barycentric', 'S3': 'left_corner'}
    rows = []
    for name in ['S1', 'S2', 'S3']:
        sc = setup_scenario(name)
        regime = sc['regime']
        rows.append({
            'scenario': name,
            'regime_computed': regime,
            'regime_expected': expected_regimes[name],
            'correct': regime == expected_regimes[name],
        })
    return pd.DataFrame(rows)


def run_dominance_equivalence() -> pd.DataFrame:
    """Verify 2D dominance condition b_1^2/B_11 > b_2^2/B_22 ↔ A_22 > A_11."""
    rows = []
    for name in ['S1', 'S2', 'S3']:
        sc = setup_scenario(name)
        result = verify_dominance_equivalence(sc['B'], sc['A'])
        result['scenario'] = name
        rows.append(result)
    return pd.DataFrame(rows)


def run_homogeneity_checks(N: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Test positive homogeneity q_alpha(cu) ≈ c * q_alpha(u) for multiple c and alpha."""
    rows = []
    sc = setup_scenario('S1')
    X = generate_lognormal_samples(sc['mu'], sc['A'], N, seed=seed)
    u_test = np.array([0.5, 0.5])
    c_values = [0.5, 2.0]
    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        results = check_homogeneity(X, u_test, alpha, c_values, tol=0.06)
        for c, info in results.items():
            rows.append({
                'alpha': alpha,
                'c': c,
                'q(cu)': round(info['q_cu'], 6),
                'c*q(u)': round(info['c_q_u'], 6),
                'rel_err': round(info['rel_err'], 6),
                'passes': info['passes'],
            })
    return pd.DataFrame(rows)


def plot_scenario_summary(output_dir: Path) -> None:
    """Plot b/beta bar chart for all three scenarios."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ['S1', 'S2', 'S3']
    x = np.arange(len(scenarios))
    b1_vals, b2_vals, beta_vals = [], [], []
    for name in scenarios:
        sc = setup_scenario(name)
        b1_vals.append(sc['b'][0])
        b2_vals.append(sc['b'][1])
        beta_vals.append(sc['beta'])

    width = 0.25
    ax.bar(x - width, b1_vals, width, label='b_1', color='steelblue')
    ax.bar(x, b2_vals, width, label='b_2', color='orange')
    ax.bar(x + width, beta_vals, width, label='beta', color='green')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Value')
    ax.set_title('Experiment 1: b_i and beta for each scenario')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'exp1_b_beta_summary.png', dpi=120)
    plt.close(fig)


def run_exp1() -> dict:
    """Run all Experiment 1 checks and return summary results."""
    print("=" * 60)
    print("Experiment 1: Sanity Checks & Parameter-Regime Validation")
    print("=" * 60)

    print("\n--- Matrix inversion checks ---")
    df_inv = run_matrix_inversion_checks()
    print(df_inv.to_string(index=False))

    print("\n--- b / beta / t* checks ---")
    df_b = run_b_beta_checks()
    print(df_b[['scenario', 'b_computed', 'b_expected', 'b_max_err', 'beta_err', 'b_ok', 'beta_ok']].to_string(index=False))

    print("\n--- Regime classification ---")
    df_reg = run_regime_classification()
    print(df_reg.to_string(index=False))

    print("\n--- Dominance equivalence (2D) ---")
    df_dom = run_dominance_equivalence()
    print(df_dom.to_string(index=False))

    print("\n--- Positive homogeneity checks (S1, N=100k) ---")
    df_hom = run_homogeneity_checks(N=100_000)
    print(df_hom.to_string(index=False))

    plot_scenario_summary(RESULTS_DIR)

    # Assert key results
    assert df_inv['near_machine_eps'].all(), "Matrix inversion not near machine eps!"
    assert df_b['b_ok'].all(), "b computation error!"
    assert df_b['beta_ok'].all(), "beta computation error!"
    assert df_reg['correct'].all(), "Regime classification failed!"
    assert df_dom['equivalent'].all(), "Dominance equivalence failed!"
    assert df_hom['passes'].all(), "Homogeneity checks failed!"

    print("\n✓ All Experiment 1 checks passed.")
    return {
        'matrix_inversion': df_inv,
        'b_beta': df_b,
        'regime': df_reg,
        'dominance': df_dom,
        'homogeneity': df_hom,
    }


if __name__ == '__main__':
    results = run_exp1()
