"""
Experiment 6: Surface-measure geometry and mode flow on pulled-back level sets.

Numerically reproduces the paper's geometric mechanism by working with the
pulled-back level-set density g_y on S_1 and tracking mass concentration.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import setup_scenario
from src.geometry import (
    lognormal_log_density_phi,
    compute_g_y,
    compute_Zy_normalization,
    compute_mu_y_moments,
    compute_mode_gy,
    compute_projected_hessian_criterion,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_mode_flow_analysis(
    scenario_name: str,
    u: np.ndarray,
    y_values: np.ndarray,
    n_x1: int = 1000,
) -> dict:
    """Track mode of g_y on S_1 as y varies.

    Parameters
    ----------
    scenario_name : str
        'S1', 'S2', or 'S3'.
    u : np.ndarray
        Portfolio weight vector (sums to 1, positive entries).
    y_values : np.ndarray
        y values to scan.
    n_x1 : int
        Grid resolution for x_1.

    Returns
    -------
    dict with mode locations, density profiles, and moment diagnostics.
    """
    sc = setup_scenario(scenario_name)
    mu, A, B, b = sc['mu'], sc['A'], sc['B'], sc['b']
    u1, u2 = float(u[0]), float(u[1])

    modes = []
    moments_list = []
    hessian_signs = []

    # Fix a reference y to build x_1 grid range
    y_mid = float(np.median(y_values))
    x1_max = y_mid / u1 * 0.999
    x1_grid = np.linspace(1e-6, x1_max, n_x1)

    for y in y_values:
        x1_lo = 1e-7
        x1_hi = (y - 1e-8) / u1  # x_2 must be positive
        if x1_hi <= x1_lo:
            modes.append(np.nan)
            moments_list.append({})
            hessian_signs.append(np.nan)
            continue

        x1_eval = np.linspace(x1_lo, x1_hi * 0.9999, n_x1)
        log_gy = compute_g_y(y, x1_eval, u, mu, B, b, log_scale=True)

        # Find mode
        mode_x1 = compute_mode_gy(y, u, mu, B, b, x1_lo, x1_hi * 0.9999)
        modes.append(mode_x1)

        # Compute moments under mu_y
        mom = compute_mu_y_moments(y, x1_eval, u, mu, B, b)
        moments_list.append(mom)

        # Hessian sign criterion
        hs = compute_projected_hessian_criterion(y, u, mu, B, b, n_x1=n_x1)
        hessian_signs.append(hs)

    return {
        'scenario': scenario_name,
        'u': u,
        'y_values': y_values,
        'modes': np.array(modes),
        'moments': moments_list,
        'hessian_signs': np.array(hessian_signs),
    }


def plot_mode_flow(res: dict, output_dir: Path) -> None:
    """Plot mode location as a function of y."""
    name = res['scenario']
    y_vals = res['y_values']
    modes = res['modes']
    hsigns = res['hessian_signs']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(y_vals, modes, 'o-', color='steelblue', markersize=3)
    ax.set_xlabel('y (level)')
    ax.set_ylabel('x_1 mode location')
    ax.set_title(f'{name}: Mode of g_y vs y')

    ax = axes[1]
    ax.plot(y_vals, hsigns, 's-', color='firebrick', markersize=3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('y (level)')
    ax.set_ylabel('Hessian sign criterion')
    ax.set_title(f'{name}: Projected Hessian sign vs y')

    plt.suptitle(f'Experiment 6: Surface geometry — {name}')
    plt.tight_layout()
    fig.savefig(output_dir / f'exp6_surface_geometry_{name}.png', dpi=120)
    plt.close(fig)


def plot_density_profiles(
    scenario_name: str,
    u: np.ndarray,
    y_samples: list,
    output_dir: Path,
    n_x1: int = 500,
) -> None:
    """Plot normalized g_y density profiles on S_1 for selected y values."""
    sc = setup_scenario(scenario_name)
    mu, A, B, b = sc['mu'], sc['A'], sc['B'], sc['b']

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_samples)))

    for i, y in enumerate(y_samples):
        u1 = float(u[0])
        x1_hi = (y - 1e-8) / u1 * 0.9999
        x1_eval = np.linspace(1e-7, x1_hi, n_x1)
        log_gy = compute_g_y(y, x1_eval, u, mu, B, b, log_scale=True)
        # Stabilize and normalize
        log_gy -= np.max(log_gy)
        gy_norm = np.exp(log_gy)
        gy_norm /= np.max(gy_norm)
        ax.plot(x1_eval, gy_norm, color=colors[i], label=f'y={y:.2f}')

    ax.set_xlabel('x_1 on S_1')
    ax.set_ylabel('g_y(x_1) / max g_y')
    ax.set_title(f'{scenario_name}: Density profile g_y on S_1 for varying y')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / f'exp6_density_profiles_{scenario_name}.png', dpi=120)
    plt.close(fig)


def run_exp6() -> dict:
    """Run Experiment 6: Surface-measure geometry for S1 and S3."""
    print("=" * 60)
    print("Experiment 6: Surface-Measure Geometry and Mode Flow")
    print("=" * 60)

    all_results = {}
    summary_rows = []

    u_test = np.array([0.5, 0.5])

    for sc_name in ['S1', 'S3']:
        sc = setup_scenario(sc_name)
        print(f"\n  Scenario {sc_name}: b={sc['b']}, regime={sc['regime']}")

        # Choose y range appropriate for the scenario
        if sc_name == 'S1':
            y_values = np.logspace(-1.5, 1.5, 25)
        else:  # S3 (corner regime)
            y_values = np.logspace(-1.5, 2.0, 25)

        res = run_mode_flow_analysis(sc_name, u_test, y_values, n_x1=500)
        plot_mode_flow(res, RESULTS_DIR)

        # Plot density profiles at selected y
        y_samples = [float(y_values[4]), float(y_values[12]), float(y_values[20])]
        plot_density_profiles(sc_name, u_test, y_samples, RESULTS_DIR, n_x1=300)

        all_results[sc_name] = res

        valid_mask = ~np.isnan(res['modes'])
        mode_range = (float(np.nanmin(res['modes'])), float(np.nanmax(res['modes'])))
        hs_valid = res['hessian_signs'][valid_mask & ~np.isnan(res['hessian_signs'])]
        hs_neg_frac = float(np.mean(hs_valid < 0)) if len(hs_valid) > 0 else np.nan

        summary_rows.append({
            'scenario': sc_name,
            'regime': sc['regime'],
            'mode_x1_min': round(mode_range[0], 6),
            'mode_x1_max': round(mode_range[1], 6),
            'hessian_neg_frac': round(hs_neg_frac, 4) if not np.isnan(hs_neg_frac) else np.nan,
        })
        print(f"    Mode range: {mode_range[0]:.4f} — {mode_range[1]:.4f}")
        print(f"    Hessian sign < 0 fraction: {hs_neg_frac:.3f}")

    df = pd.DataFrame(summary_rows)
    print("\nSummary:")
    print(df.to_string(index=False))

    print("\n✓ Experiment 6 complete: mode flow and hessian criterion plots generated.")
    return {'summary': df, 'all_results': all_results}


if __name__ == '__main__':
    run_exp6()
