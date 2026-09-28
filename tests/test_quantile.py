"""
Unit tests for src/quantile.py.

Tests cover:
- empirical_quantile_left: inf-definition correctness, edge cases
- empirical_quantile_curve: shape and monotonicity
- compute_curvature: finite-difference second derivative
- gaussian_quantile_on_simplex: analytic formula
"""
import numpy as np
import pytest

from src.quantile import (
    empirical_quantile_left,
    empirical_quantile_curve,
    compute_curvature,
    gaussian_quantile_on_simplex,
)


# ---------------------------------------------------------------------------
# empirical_quantile_left
# ---------------------------------------------------------------------------

class TestEmpiricalQuantileLeft:
    def test_sorted_array_alpha_half(self):
        Y = np.arange(1.0, 101.0)   # 1..100
        # alpha=0.5 → idx = ceil(50) - 1 = 49 → Y[49] = 50.0
        assert empirical_quantile_left(Y, 0.5) == 50.0

    def test_alpha_one_gives_maximum(self):
        Y = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
        assert empirical_quantile_left(Y, 1.0) == pytest.approx(9.0)

    def test_alpha_near_zero_gives_minimum(self):
        Y = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
        # alpha → very small, idx = ceil(0) - 1 clipped to 0 → minimum
        assert empirical_quantile_left(Y, 1e-9) == pytest.approx(1.0)

    def test_left_continuous_inf_definition(self):
        # For N=10, alpha=0.3: idx = ceil(3.0) - 1 = 2 (0-indexed)
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = empirical_quantile_left(Y, 0.3)
        assert result == pytest.approx(3.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='empty'):
            empirical_quantile_left(np.array([]), 0.5)

    def test_single_element(self):
        Y = np.array([7.0])
        assert empirical_quantile_left(Y, 0.1) == 7.0
        assert empirical_quantile_left(Y, 0.9) == 7.0

    def test_result_in_sample_range(self):
        rng = np.random.default_rng(0)
        Y = rng.exponential(2.0, size=10000)
        for alpha in [0.01, 0.1, 0.5, 0.9, 0.99]:
            q = empirical_quantile_left(Y, alpha)
            assert Y.min() <= q <= Y.max()

    def test_monotone_in_alpha(self):
        rng = np.random.default_rng(1)
        Y = rng.normal(0, 1, 10000)
        alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
        qs = [empirical_quantile_left(Y, a) for a in alphas]
        for i in range(len(qs) - 1):
            assert qs[i] <= qs[i + 1]


# ---------------------------------------------------------------------------
# empirical_quantile_curve
# ---------------------------------------------------------------------------

class TestEmpiricalQuantileCurve:
    def setup_method(self):
        rng = np.random.default_rng(42)
        # Simple bivariate lognormal
        self.X = np.exp(rng.multivariate_normal([0, 0], np.eye(2), size=50_000))
        self.t_grid = np.linspace(0.0, 1.0, 51)

    def test_output_shape(self):
        q = empirical_quantile_curve(self.X, self.t_grid, 0.5)
        assert q.shape == (51,)

    def test_endpoints_pure_asset(self):
        # t=0 → Y = X_2; t=1 → Y = X_1
        q = empirical_quantile_curve(self.X, np.array([0.0, 1.0]), 0.5)
        q0_direct = empirical_quantile_left(self.X[:, 1], 0.5)
        q1_direct = empirical_quantile_left(self.X[:, 0], 0.5)
        assert q[0] == pytest.approx(q0_direct)
        assert q[1] == pytest.approx(q1_direct)

    def test_positive_values_all_positive(self):
        q = empirical_quantile_curve(self.X, self.t_grid, 0.1)
        assert np.all(q > 0)


# ---------------------------------------------------------------------------
# compute_curvature
# ---------------------------------------------------------------------------

class TestComputeCurvature:
    def test_linear_function_zero_curvature(self):
        t = np.linspace(0, 1, 101)
        q = 3.0 * t + 1.0   # linear → zero second derivative
        dt = t[1] - t[0]
        curv = compute_curvature(q, dt)
        np.testing.assert_allclose(curv, 0.0, atol=1e-8)

    def test_quadratic_function_constant_curvature(self):
        t = np.linspace(0, 1, 101)
        q = 4.0 * t ** 2    # d^2/dt^2 = 8
        dt = t[1] - t[0]
        curv = compute_curvature(q, dt)
        np.testing.assert_allclose(curv, 8.0, atol=1e-8)

    def test_concave_function_negative_curvature(self):
        t = np.linspace(0.01, 0.99, 101)
        q = np.sqrt(t * (1.0 - t))   # concave
        dt = t[1] - t[0]
        curv = compute_curvature(q, dt)
        assert np.all(curv < 0)

    def test_output_length(self):
        q = np.ones(50)
        dt = 0.01
        assert len(compute_curvature(q, dt)) == 48


# ---------------------------------------------------------------------------
# gaussian_quantile_on_simplex
# ---------------------------------------------------------------------------

class TestGaussianQuantileOnSimplex:
    def test_diagonal_case_left_tail(self):
        # For uncorrelated equal-variance case at t=0.5: Y~N(0, A11/4+A22/4)
        A = np.eye(2)
        mu = np.zeros(2)
        t_grid = np.array([0.5])
        from scipy.stats import norm
        alpha = 0.1
        q = gaussian_quantile_on_simplex(A, mu, alpha, t_grid)
        # Y = 0.5 X_1 + 0.5 X_2, variance = 0.25 + 0.25 = 0.5
        expected = norm.ppf(alpha) * np.sqrt(0.5)
        assert abs(q[0] - expected) < 1e-12

    def test_left_tail_negative_right_tail_positive(self):
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        mu = np.zeros(2)
        t_grid = np.linspace(0.1, 0.9, 50)
        q_left = gaussian_quantile_on_simplex(A, mu, 0.1, t_grid)
        q_right = gaussian_quantile_on_simplex(A, mu, 0.9, t_grid)
        assert np.all(q_left < 0)
        assert np.all(q_right > 0)

    def test_curvature_left_concave_right_convex(self):
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        mu = np.zeros(2)
        t_grid = np.linspace(0.0, 1.0, 201)
        dt = t_grid[1] - t_grid[0]

        q_left = gaussian_quantile_on_simplex(A, mu, 0.1, t_grid)
        q_right = gaussian_quantile_on_simplex(A, mu, 0.9, t_grid)

        curv_left = compute_curvature(q_left, dt)
        curv_right = compute_curvature(q_right, dt)

        # Left tail: concave → mostly negative curvature
        assert np.mean(curv_left < 0) > 0.9, "Left-tail should be concave"
        # Right tail: convex → mostly positive curvature
        assert np.mean(curv_right > 0) > 0.9, "Right-tail should be convex"
