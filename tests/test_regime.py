"""
Unit tests for src/regime.py and src/utils.py.

Tests cover:
- Dominance equivalence verification
- Theoretical normalized shape (barycentric formula)
- Section 5 optimizer formula in terms of A
- Simplex grid generation
- Normalized shape, optimizer, and error metrics
- Homogeneity checks
"""
import numpy as np
import pytest

from src.model import setup_scenario, compute_b_beta
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


# ---------------------------------------------------------------------------
# verify_dominance_equivalence
# ---------------------------------------------------------------------------

class TestDominanceEquivalence:
    def test_S1_equivalent(self):
        sc = setup_scenario('S1')
        result = verify_dominance_equivalence(sc['B'], sc['A'])
        assert result['equivalent']

    def test_S2_equivalent(self):
        sc = setup_scenario('S2')
        result = verify_dominance_equivalence(sc['B'], sc['A'])
        assert result['equivalent']

    def test_S3_equivalent(self):
        sc = setup_scenario('S3')
        result = verify_dominance_equivalence(sc['B'], sc['A'])
        assert result['equivalent']

    def test_returns_numeric_fields(self):
        sc = setup_scenario('S1')
        result = verify_dominance_equivalence(sc['B'], sc['A'])
        for key in ['lhs', 'rhs', 'b1sq_over_B11', 'b2sq_over_B22', 'A11', 'A22']:
            assert key in result


# ---------------------------------------------------------------------------
# theoretical_normalized_shape_barycentric
# ---------------------------------------------------------------------------

class TestBarycentricShape:
    def test_maximum_at_t_star(self):
        sc = setup_scenario('S1')
        b, beta = sc['b'], sc['beta']
        t_grid = np.linspace(0.001, 0.999, 1001)
        q0 = theoretical_normalized_shape_barycentric(t_grid, b, beta)
        t_max = t_grid[np.argmax(q0)]
        t_star = b[0] / beta
        assert abs(t_max - t_star) < 0.002

    def test_shape_bounded_0_1(self):
        sc = setup_scenario('S1')
        t_grid = np.linspace(0.001, 0.999, 501)
        q0 = theoretical_normalized_shape_barycentric(t_grid, sc['b'], sc['beta'])
        assert np.all(q0 >= 0)
        assert np.max(q0) <= 1.0 + 1e-10

    def test_shape_values_at_endpoints_near_zero(self):
        sc = setup_scenario('S1')
        t_grid = np.array([0.001, 0.999])
        q0 = theoretical_normalized_shape_barycentric(t_grid, sc['b'], sc['beta'])
        assert q0[0] < 0.3
        assert q0[1] < 0.3


# ---------------------------------------------------------------------------
# section5_optimizer_formula
# ---------------------------------------------------------------------------

class TestSection5Optimizer:
    def test_matches_b_over_beta_S1(self):
        sc = setup_scenario('S1')
        u_sec5 = section5_optimizer_formula(sc['A'])
        u_bbeta = sc['b'] / sc['beta']
        np.testing.assert_allclose(u_sec5, u_bbeta, atol=1e-12)

    def test_matches_b_over_beta_S2(self):
        sc = setup_scenario('S2')
        u_sec5 = section5_optimizer_formula(sc['A'])
        u_bbeta = sc['b'] / sc['beta']
        np.testing.assert_allclose(u_sec5, u_bbeta, atol=1e-12)

    def test_sums_to_one(self):
        for name in ['S1', 'S2']:
            sc = setup_scenario(name)
            u = section5_optimizer_formula(sc['A'])
            assert abs(u.sum() - 1.0) < 1e-12

    def test_both_components_positive_barycentric(self):
        for name in ['S1', 'S2']:
            sc = setup_scenario(name)
            u = section5_optimizer_formula(sc['A'])
            assert np.all(u > 0)


# ---------------------------------------------------------------------------
# simplex_grid
# ---------------------------------------------------------------------------

class TestSimplexGrid:
    def test_default_length(self):
        g = simplex_grid()
        assert len(g) == 201

    def test_endpoints(self):
        g = simplex_grid(101)
        assert g[0] == pytest.approx(0.0)
        assert g[-1] == pytest.approx(1.0)

    def test_uniform_spacing(self):
        g = simplex_grid(51)
        diffs = np.diff(g)
        assert np.allclose(diffs, diffs[0])


# ---------------------------------------------------------------------------
# normalized_shape
# ---------------------------------------------------------------------------

class TestNormalizedShape:
    def test_max_becomes_one(self):
        q = np.array([1.0, 3.0, 2.0])
        t = np.array([0.0, 0.5, 1.0])
        q_norm = normalized_shape(q, t)
        assert q_norm.max() == pytest.approx(1.0)

    def test_provided_k_alpha(self):
        q = np.array([2.0, 4.0, 3.0])
        t = np.array([0.0, 0.5, 1.0])
        q_norm = normalized_shape(q, t, k_alpha=4.0)
        np.testing.assert_allclose(q_norm, [0.5, 1.0, 0.75])

    def test_zero_k_alpha_gives_zeros(self):
        q = np.ones(5)
        t = np.linspace(0, 1, 5)
        result = normalized_shape(q, t, k_alpha=0.0)
        np.testing.assert_array_equal(result, np.zeros(5))


# ---------------------------------------------------------------------------
# find_optimizer
# ---------------------------------------------------------------------------

class TestFindOptimizer:
    def test_finds_correct_t(self):
        t = np.linspace(0, 1, 101)
        q = -(t - 0.4) ** 2 + 1.0   # maximum at t=0.4
        t_star = find_optimizer(q, t)
        assert abs(t_star - 0.4) < 0.02


# ---------------------------------------------------------------------------
# shape_error_metrics
# ---------------------------------------------------------------------------

class TestShapeErrorMetrics:
    def test_identical_curves_zero_error(self):
        t = np.linspace(0, 1, 101)
        q = np.sin(np.pi * t) + 1.0
        metrics = shape_error_metrics(q, q, t)
        assert metrics['sup_norm'] < 1e-12
        assert metrics['sup_rel'] < 1e-12

    def test_known_offset(self):
        t = np.linspace(0, 1, 101)
        q_theory = np.ones(101) * 2.0
        q_norm = np.ones(101) * 2.1
        metrics = shape_error_metrics(q_norm, q_theory, t)
        assert abs(metrics['sup_norm'] - 0.1) < 1e-10


# ---------------------------------------------------------------------------
# check_homogeneity
# ---------------------------------------------------------------------------

class TestCheckHomogeneity:
    def setup_method(self):
        sc = setup_scenario('S1')
        self.X = np.exp(
            np.random.default_rng(42).multivariate_normal(
                sc['mu'], sc['A'], size=100_000
            )
        )

    def test_passes_c1(self):
        u = np.array([0.5, 0.5])
        results = check_homogeneity(self.X, u, 0.5, [1.0])
        # c=1 → trivially passes
        assert results[1.0]['passes']

    def test_homogeneity_with_c2(self):
        u = np.array([0.5, 0.5])
        results = check_homogeneity(self.X, u, 0.5, [2.0])
        # For N=100k, relative error should be well within 5%
        assert results[2.0]['rel_err'] < 0.05

    def test_homogeneity_with_c_half(self):
        u = np.array([0.5, 0.5])
        results = check_homogeneity(self.X, u, 0.5, [0.5])
        assert results[0.5]['rel_err'] < 0.05
