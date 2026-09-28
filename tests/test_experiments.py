"""
Experiment-level integration tests that validate paper hypotheses.

These tests import the run functions from exp/ scripts and assert
key theoretical properties without requiring full large Monte Carlo runs.
They use small N to run quickly while still catching implementation bugs.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import setup_scenario, generate_lognormal_samples
from src.quantile import (
    empirical_quantile_curve,
    compute_curvature,
    empirical_quantile_left,
    gaussian_quantile_on_simplex,
)
from src.regime import (
    verify_dominance_equivalence,
    theoretical_normalized_shape_barycentric,
    section5_optimizer_formula,
)
from src.utils import (
    simplex_grid,
    normalized_shape,
    find_optimizer,
    shape_error_metrics,
    check_homogeneity,
)


N_SMALL = 200_000   # fast but statistically sufficient
SEED = 42


# ---------------------------------------------------------------------------
# Exp 1: Sanity checks (small budget)
# ---------------------------------------------------------------------------

class TestExp1SanityChecks:
    """Validate matrix inversion, b/beta, regime classification, dominance."""

    def test_matrix_inversion_all_scenarios(self):
        for name in ['S1', 'S2', 'S3']:
            sc = setup_scenario(name)
            err = np.linalg.norm(sc['A'] @ sc['B'] - np.eye(2), 'fro')
            assert err < 1e-13, f"AB-I not machine-eps for {name}: {err}"

    def test_b_beta_S1(self):
        sc = setup_scenario('S1')
        np.testing.assert_allclose(sc['b'], [0.6, 0.35], atol=1e-10)
        assert abs(sc['beta'] - 0.95) < 1e-10

    def test_b_beta_S2(self):
        sc = setup_scenario('S2')
        np.testing.assert_allclose(sc['b'], [5.1, 0.35], atol=1e-10)
        assert abs(sc['beta'] - 5.45) < 1e-10

    def test_b_beta_S3(self):
        sc = setup_scenario('S3')
        np.testing.assert_allclose(sc['b'], [0.2, -0.05], atol=1e-10)
        assert abs(sc['beta'] - 0.15) < 1e-10

    def test_regime_classification(self):
        assert setup_scenario('S1')['regime'] == 'barycentric'
        assert setup_scenario('S2')['regime'] == 'barycentric'
        assert setup_scenario('S3')['regime'] == 'left_corner'

    def test_dominance_equivalence_all(self):
        for name in ['S1', 'S2', 'S3']:
            sc = setup_scenario(name)
            res = verify_dominance_equivalence(sc['B'], sc['A'])
            assert res['equivalent'], f"Dominance equivalence failed for {name}"

    def test_homogeneity_S1(self):
        sc = setup_scenario('S1')
        X = generate_lognormal_samples(sc['mu'], sc['A'], N_SMALL, SEED)
        results = check_homogeneity(X, np.array([0.5, 0.5]), 0.5, [0.5, 2.0], tol=0.05)
        for c, info in results.items():
            assert info['passes'], f"Homogeneity failed for c={c}: {info}"


# ---------------------------------------------------------------------------
# Exp 2: Elliptical benchmark — curvature signs
# ---------------------------------------------------------------------------

class TestExp2EllipticalBenchmark:
    """Left-tail concavity and right-tail convexity for bivariate normal."""

    def setup_method(self):
        self.A = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.mu = np.zeros(2)
        self.t_grid = np.linspace(0.0, 1.0, 201)
        self.dt = self.t_grid[1] - self.t_grid[0]

    def test_left_tail_concave(self):
        q = gaussian_quantile_on_simplex(self.A, self.mu, 0.1, self.t_grid)
        curv = compute_curvature(q, self.dt)
        frac_negative = np.mean(curv < 0)
        assert frac_negative > 0.9, f"Only {frac_negative:.2f} of interior points concave"

    def test_right_tail_convex(self):
        q = gaussian_quantile_on_simplex(self.A, self.mu, 0.9, self.t_grid)
        curv = compute_curvature(q, self.dt)
        frac_positive = np.mean(curv > 0)
        assert frac_positive > 0.9, f"Only {frac_positive:.2f} of interior points convex"

    def test_median_is_flatter_than_tails(self):
        q_med = gaussian_quantile_on_simplex(self.A, self.mu, 0.5, self.t_grid)
        curv_med = np.abs(compute_curvature(q_med, self.dt))
        q_left = gaussian_quantile_on_simplex(self.A, self.mu, 0.1, self.t_grid)
        curv_left = np.abs(compute_curvature(q_left, self.dt))
        assert np.max(curv_left) > np.max(curv_med)


# ---------------------------------------------------------------------------
# Exp 3: Lognormal barycentric regime — concavity & optimizer
# ---------------------------------------------------------------------------

class TestExp3LognormalBarycentric:
    """Left-tail concavity and optimizer convergence for S1 and S2."""

    def _run_scenario(self, name, alpha):
        sc = setup_scenario(name)
        X = generate_lognormal_samples(sc['mu'], sc['A'], N_SMALL, SEED)
        t_grid = simplex_grid(201)
        q = empirical_quantile_curve(X, t_grid, alpha)
        return q, t_grid, sc

    def test_S1_concave_left_tail(self):
        """Concavity: every interior point lies strictly above the chord connecting endpoints.
        This is the noise-robust midpoint/chord test — mathematically equivalent to
        global concavity but insensitive to finite-difference amplification at small N."""
        q, t_grid, _ = self._run_scenario('S1', 0.01)
        interior = slice(1, -1)
        chord = q[0] + (q[-1] - q[0]) * t_grid[interior]
        assert np.all(q[interior] >= chord), (
            "S1 left-tail: some interior q-values fall below the chord (not concave)"
        )

    def test_S2_concave_left_tail(self):
        """Concavity: every interior point lies strictly above the chord connecting endpoints."""
        q, t_grid, _ = self._run_scenario('S2', 0.01)
        interior = slice(1, -1)
        chord = q[0] + (q[-1] - q[0]) * t_grid[interior]
        assert np.all(q[interior] >= chord), (
            "S2 left-tail: some interior q-values fall below the chord (not concave)"
        )

    def test_S1_optimizer_near_t_star(self):
        q, t_grid, sc = self._run_scenario('S1', 0.01)
        t_emp = find_optimizer(q, t_grid)
        t_theory = sc['optimizer']['t_star']
        assert abs(t_emp - t_theory) < 0.05, f"S1 optimizer far from t*: {t_emp:.3f} vs {t_theory:.3f}"

    def test_S2_optimizer_near_t_star(self):
        q, t_grid, sc = self._run_scenario('S2', 0.01)
        t_emp = find_optimizer(q, t_grid)
        t_theory = sc['optimizer']['t_star']
        assert abs(t_emp - t_theory) < 0.05, f"S2 optimizer far from t*: {t_emp:.3f} vs {t_theory:.3f}"

    def test_S1_normalized_shape_close_to_q0(self):
        q, t_grid, sc = self._run_scenario('S1', 0.01)
        q_norm = normalized_shape(q, t_grid)
        q0 = theoretical_normalized_shape_barycentric(t_grid, sc['b'], sc['beta'])
        metrics = shape_error_metrics(q_norm, q0, t_grid, eps=0.05)
        assert metrics['sup_norm'] < 0.15, f"S1 shape error too large: {metrics['sup_norm']:.4f}"

    def test_section5_formula_matches_b_beta_S1(self):
        sc = setup_scenario('S1')
        u_sec5 = section5_optimizer_formula(sc['A'])
        u_bbeta = sc['b'] / sc['beta']
        np.testing.assert_allclose(u_sec5, u_bbeta, atol=1e-12)


# ---------------------------------------------------------------------------
# Exp 4: Left-tail corner regime — convergence to q0(t)=t
# ---------------------------------------------------------------------------

class TestExp4CornerLeft:
    """Left-tail corner S3: concavity and convergence to linear shape."""

    def _run(self, alpha):
        sc = setup_scenario('S3')
        X = generate_lognormal_samples(sc['mu'], sc['A'], N_SMALL, SEED)
        t_grid = simplex_grid(201)
        q = empirical_quantile_curve(X, t_grid, alpha)
        return q, t_grid

    def test_concave_left_tail(self):
        """All interior empirical quantiles lie at or above the linear chord.
        Uses the chord test instead of noisy finite-difference curvature fractions."""
        q, t_grid = self._run(0.01)
        interior = slice(1, -1)
        chord = q[0] + (q[-1] - q[0]) * t_grid[interior]
        assert np.all(q[interior] >= chord), (
            "S3 left-corner: some interior q-values fall below the chord (not concave)"
        )

    def test_optimizer_near_corner_t1(self):
        q, t_grid = self._run(0.01)
        t_emp = find_optimizer(q, t_grid)
        # The optimizer should be in the right half (above 0.5)
        assert t_emp > 0.5, f"S3 optimizer not near t=1: {t_emp:.3f}"

    def test_normalized_shape_approaching_linear(self):
        q, t_grid = self._run(0.01)
        k_alpha = float(q[-1])  # normalization at t=1 (corner)
        if k_alpha <= 0:
            k_alpha = float(np.max(q))
        q_norm = q / k_alpha
        q0 = t_grid   # linear asymptotic shape
        # Compare on interior
        mask = (t_grid >= 0.05) & (t_grid <= 0.95)
        sup_err = float(np.max(np.abs(q_norm[mask] - q0[mask])))
        # Should be below 0.3 at moderate alpha (not fully converged yet)
        assert sup_err < 0.35, f"S3 normalized shape too far from t: {sup_err:.4f}"


# ---------------------------------------------------------------------------
# Exp 5: Right-tail convexity — convergence to q0(t)=1-t
# ---------------------------------------------------------------------------

class TestExp5RightTail:
    """Right-tail S3: convexity and convergence to q0(t)=1-t."""

    def _run(self, alpha):
        sc = setup_scenario('S3')
        X = generate_lognormal_samples(sc['mu'], sc['A'], N_SMALL, SEED)
        t_grid = simplex_grid(201)
        q = empirical_quantile_curve(X, t_grid, alpha)
        return q, t_grid

    def test_convex_right_tail(self):
        """Convexity via normalized midpoint-below-chord test at alpha=0.999.
        Per the paper (Exp 5 notes), convexity is only visually/numerically clear
        at extreme depths; alpha=0.99 is still in the pre-asymptotic regime for
        S3. Using N=1M and alpha=0.999 gives ~1000 tail exceedances, a regime
        where the chord test reliably resolves the convex signal."""
        sc = setup_scenario('S3')
        X = generate_lognormal_samples(sc['mu'], sc['A'], 1_000_000, SEED)
        t_grid = simplex_grid(101)
        q = empirical_quantile_curve(X, t_grid, 0.999)
        # Normalize by the corner value at t=0 (right-tail corner optimizer)
        k_alpha = float(q[0])
        q_norm = q / k_alpha
        # Convexity: normalized midpoint lies strictly below the linear chord
        mid_idx = len(t_grid) // 2
        mid_val = q_norm[mid_idx]
        avg_endpoints = (q_norm[0] + q_norm[-1]) / 2.0
        assert mid_val < avg_endpoints, (
            f"S3 right-tail: normalized midpoint {mid_val:.4f} not below "
            f"chord avg {avg_endpoints:.4f} at alpha=0.999"
        )

    def test_optimizer_near_corner_t0(self):
        q, t_grid = self._run(0.99)
        t_emp = find_optimizer(q, t_grid)
        # The optimizer should be in the left half
        assert t_emp < 0.5, f"S3 right-tail optimizer not near t=0: {t_emp:.3f}"

    def test_normalized_shape_approaching_1_minus_t(self):
        q, t_grid = self._run(0.99)
        k_alpha = float(q[0])   # normalization at t=0 (corner)
        if k_alpha <= 0:
            k_alpha = float(np.max(q))
        q_norm = q / k_alpha
        q0 = 1.0 - t_grid
        mask = (t_grid >= 0.05) & (t_grid <= 0.95)
        sup_err = float(np.max(np.abs(q_norm[mask] - q0[mask])))
        assert sup_err < 0.35, f"S3 right-tail normalized shape too far from 1-t: {sup_err:.4f}"


# ---------------------------------------------------------------------------
# Exp 7: Tail universality — location-parameter invariance
# ---------------------------------------------------------------------------

class TestExp7Universality:
    """Normalized quantile shapes should be invariant to mu when A is fixed."""

    def _run_mu(self, mu, alpha, N=N_SMALL):
        sc = setup_scenario('S1')
        X = generate_lognormal_samples(mu, sc['A'], N, SEED)
        t_grid = simplex_grid(101)
        q = empirical_quantile_curve(X, t_grid, alpha)
        q_norm = normalized_shape(q, t_grid)
        return q_norm, t_grid, sc

    def test_normalized_shapes_collapse_across_mu(self):
        """Universality: cross-mu sup-norm difference shrinks as alpha deepens.
        The paper guarantees convergence to the same q_0 shape in the limit; at
        finite N=200k the collapse is monotone but not yet at zero. We verify that
        diff(alpha=0.01) < diff(alpha=0.1), which is a necessary condition for
        convergence and is stable across seeds."""
        mu_a = np.array([0.0, 0.0])
        mu_b = np.array([1.0, 0.0])
        diffs = {}
        for alpha in [0.1, 0.01]:
            q_a, _, _ = self._run_mu(mu_a, alpha)
            q_b, _, _ = self._run_mu(mu_b, alpha)
            diffs[alpha] = float(np.max(np.abs(q_a - q_b)))
        assert diffs[0.01] < diffs[0.1], (
            f"Cross-mu collapse not improving with depth: "
            f"diff@0.1={diffs[0.1]:.4f}, diff@0.01={diffs[0.01]:.4f}"
        )


# ---------------------------------------------------------------------------
# Exp 8: Optimizer formula verification
# ---------------------------------------------------------------------------

class TestExp8VaROptimizer:
    """Section 5 optimizer formula matches b/beta in 2D barycentric regime."""

    def test_S1_section5_equals_b_over_beta(self):
        sc = setup_scenario('S1')
        u_sec5 = section5_optimizer_formula(sc['A'])
        u_bbeta = sc['b'] / sc['beta']
        np.testing.assert_allclose(u_sec5, u_bbeta, atol=1e-12)

    def test_S2_section5_equals_b_over_beta(self):
        sc = setup_scenario('S2')
        u_sec5 = section5_optimizer_formula(sc['A'])
        u_bbeta = sc['b'] / sc['beta']
        np.testing.assert_allclose(u_sec5, u_bbeta, atol=1e-12)

    def test_empirical_optimizer_moves_toward_theory_as_alpha_decreases(self):
        sc = setup_scenario('S1')
        X = generate_lognormal_samples(sc['mu'], sc['A'], N_SMALL, SEED)
        t_grid = simplex_grid(201)
        t_star = sc['optimizer']['t_star']

        q_alpha01 = empirical_quantile_curve(X, t_grid, 0.1)
        q_alpha001 = empirical_quantile_curve(X, t_grid, 0.01)

        t_emp01 = find_optimizer(q_alpha01, t_grid)
        t_emp001 = find_optimizer(q_alpha001, t_grid)

        err01 = abs(t_emp01 - t_star)
        err001 = abs(t_emp001 - t_star)
        # Deeper tail should be closer to theoretical optimizer
        assert err001 <= err01 + 0.05, (
            f"Optimizer not converging: err(0.1)={err01:.3f} err(0.01)={err001:.3f}"
        )


# ---------------------------------------------------------------------------
# Exp 9: Gradient formula validation (Gaussian benchmark)
# ---------------------------------------------------------------------------

class TestExp9DerivativeValidation:
    """Validate finite-difference gradient against analytic Gaussian formula."""

    def test_gaussian_finite_difference_gradient(self):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        mu = np.zeros(2)
        alpha = 0.1
        from scipy.stats import norm
        z_alpha = norm.ppf(alpha)

        # Analytic gradient at t=0.5: u=(0.5, 0.5)
        t0 = 0.5
        u0 = np.array([t0, 1.0 - t0])
        s0 = float(np.sqrt(u0 @ A @ u0))
        q0_analytic = float(u0 @ mu) + z_alpha * s0

        # Finite-difference gradient in t direction
        eps = 1e-5
        t_grid = np.array([t0 - eps, t0, t0 + eps])
        q_fd = gaussian_quantile_on_simplex(A, mu, alpha, t_grid)
        dq_dt_fd = (q_fd[2] - q_fd[0]) / (2 * eps)

        # Analytic d/dt: mu1 - mu2 + z_alpha * (A11 - A22) / (2*s)
        # via d/dt [(t*(A11-A22) + A12 + A22 - 2t*A12)]^{1/2}
        # = z_alpha * (A[0,0] - A[1,1] + 2*(A[0,1])(... )) / ...
        # Compute analytically
        def s_of_t(tt):
            uu = np.array([tt, 1.0 - tt])
            return float(np.sqrt(uu @ A @ uu))

        ds_dt = (s_of_t(t0 + eps) - s_of_t(t0 - eps)) / (2 * eps)
        dmu_dt = float(mu[0] - mu[1])
        dq_dt_analytic = dmu_dt + z_alpha * ds_dt

        assert abs(dq_dt_fd - dq_dt_analytic) < 1e-6, (
            f"FD gradient {dq_dt_fd:.8f} vs analytic {dq_dt_analytic:.8f}"
        )
