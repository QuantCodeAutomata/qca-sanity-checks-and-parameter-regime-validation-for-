"""
Unit tests for src/model.py core primitives.

Tests cover:
- SPD validation
- Matrix inversion consistency (A = B^{-1})
- b and beta computation against paper's exact values
- Regime classification
- Theoretical optimizer targets
- Lognormal sample generation
"""
import numpy as np
import pytest
from numpy.linalg import LinAlgError

from src.model import (
    validate_spd, compute_A_B, compute_b_beta, classify_regime,
    theoretical_optimizer, generate_lognormal_samples, setup_scenario,
    SCENARIOS,
)


# ---------------------------------------------------------------------------
# validate_spd
# ---------------------------------------------------------------------------

class TestValidateSPD:
    def test_identity_is_spd(self):
        assert validate_spd(np.eye(2)) is True

    def test_paper_S1_B_is_spd(self):
        B = np.array([[0.5, 0.1], [0.1, 0.25]])
        assert validate_spd(B) is True

    def test_not_symmetric_raises(self):
        M = np.array([[1.0, 0.5], [0.1, 1.0]])
        with pytest.raises(ValueError, match='symmetric'):
            validate_spd(M)

    def test_indefinite_raises(self):
        M = np.array([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError):
            validate_spd(M)

    def test_nearly_singular_symmetric_spd(self):
        M = np.array([[1e-8, 0.0], [0.0, 1e-8]])
        assert validate_spd(M) is True


# ---------------------------------------------------------------------------
# compute_A_B
# ---------------------------------------------------------------------------

class TestComputeAB:
    def test_B_input_gives_correct_A(self):
        B = np.array([[0.5, 0.1], [0.1, 0.25]])
        mu, A, B_out = compute_A_B({'mu': [0.0, 0.0], 'B': B})
        assert np.allclose(A @ B, np.eye(2), atol=1e-13)
        assert np.allclose(B_out, B)

    def test_A_input_gives_correct_B(self):
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        mu, A_out, B = compute_A_B({'mu': [0.0, 0.0], 'A': A})
        assert np.allclose(B @ A, np.eye(2), atol=1e-13)
        assert np.allclose(A_out, A)

    def test_missing_A_B_raises(self):
        with pytest.raises(ValueError, match='B .* A'):
            compute_A_B({'mu': [0.0, 0.0]})

    def test_inversion_error_all_scenarios(self):
        for name in ['S1', 'S2', 'S3']:
            params = SCENARIOS[name]
            mu, A, B = compute_A_B(params)
            assert np.linalg.norm(A @ B - np.eye(2), 'fro') < 1e-13
            assert np.linalg.norm(B @ A - np.eye(2), 'fro') < 1e-13


# ---------------------------------------------------------------------------
# compute_b_beta — paper exact values
# ---------------------------------------------------------------------------

class TestComputeBBeta:
    def test_S1_exact_values(self):
        sc = setup_scenario('S1')
        b, beta = sc['b'], sc['beta']
        np.testing.assert_allclose(b, [0.6, 0.35], atol=1e-10)
        assert abs(beta - 0.95) < 1e-10

    def test_S2_exact_values(self):
        sc = setup_scenario('S2')
        b, beta = sc['b'], sc['beta']
        np.testing.assert_allclose(b, [5.1, 0.35], atol=1e-10)
        assert abs(beta - 5.45) < 1e-10

    def test_S3_exact_values(self):
        sc = setup_scenario('S3')
        b, beta = sc['b'], sc['beta']
        np.testing.assert_allclose(b, [0.2, -0.05], atol=1e-10)
        assert abs(beta - 0.15) < 1e-10

    def test_beta_equals_sum_b(self):
        for name in ['S1', 'S2', 'S3']:
            sc = setup_scenario(name)
            assert abs(sc['beta'] - sc['b'].sum()) < 1e-12

    def test_b_is_column_sum_of_B(self):
        sc = setup_scenario('S1')
        b_expected = sc['B'].sum(axis=0)
        np.testing.assert_allclose(sc['b'], b_expected, atol=1e-14)


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_S1_barycentric(self):
        sc = setup_scenario('S1')
        assert sc['regime'] == 'barycentric'

    def test_S2_barycentric(self):
        sc = setup_scenario('S2')
        assert sc['regime'] == 'barycentric'

    def test_S3_left_corner(self):
        sc = setup_scenario('S3')
        assert sc['regime'] == 'left_corner'

    def test_all_negative_b_gives_right_corner(self):
        # Manufacture a matrix where b_1 < 0, b_2 > 0 → right_corner
        B = np.array([[0.25, -0.3], [-0.3, 0.5]])
        A = np.linalg.inv(B)
        assert classify_regime(B, A) == 'right_corner'


# ---------------------------------------------------------------------------
# theoretical_optimizer
# ---------------------------------------------------------------------------

class TestTheoreticalOptimizer:
    def test_S1_t_star(self):
        sc = setup_scenario('S1')
        t_star = sc['optimizer']['t_star']
        assert abs(t_star - 0.6315789473684210) < 1e-10

    def test_S2_t_star(self):
        sc = setup_scenario('S2')
        t_star = sc['optimizer']['t_star']
        assert abs(t_star - 0.9357798165137615) < 1e-10

    def test_S3_left_corner_optimizer(self):
        sc = setup_scenario('S3')
        np.testing.assert_array_equal(sc['optimizer']['u_star'], [1.0, 0.0])
        assert sc['optimizer']['t_star'] == 1.0

    def test_barycentric_u_star_on_simplex(self):
        for name in ['S1', 'S2']:
            sc = setup_scenario(name)
            u_star = sc['optimizer']['u_star']
            assert abs(u_star.sum() - 1.0) < 1e-12
            assert np.all(u_star > 0)

    def test_barycentric_u_star_equals_b_over_beta(self):
        for name in ['S1', 'S2']:
            sc = setup_scenario(name)
            b, beta = sc['b'], sc['beta']
            np.testing.assert_allclose(sc['optimizer']['u_star'], b / beta, atol=1e-12)


# ---------------------------------------------------------------------------
# generate_lognormal_samples
# ---------------------------------------------------------------------------

class TestGenerateLognormalSamples:
    def test_output_shape(self):
        mu = np.array([0.0, 0.0])
        A = np.eye(2)
        X = generate_lognormal_samples(mu, A, N=1000, seed=0)
        assert X.shape == (1000, 2)

    def test_all_positive(self):
        mu = np.array([0.0, 0.0])
        A = np.eye(2)
        X = generate_lognormal_samples(mu, A, N=5000, seed=1)
        assert np.all(X > 0)

    def test_log_normal_distributed(self):
        # log(X_i) should have mean ≈ mu_i and std close to sqrt(A_ii)
        mu = np.array([1.0, -0.5])
        A = np.diag([0.5, 0.25])
        X = generate_lognormal_samples(mu, A, N=200_000, seed=42)
        log_X = np.log(X)
        np.testing.assert_allclose(log_X.mean(axis=0), mu, atol=0.02)
        np.testing.assert_allclose(log_X.var(axis=0), np.diag(A), atol=0.02)

    def test_reproducibility(self):
        mu = np.array([0.0, 0.0])
        A = np.eye(2)
        X1 = generate_lognormal_samples(mu, A, 500, seed=99)
        X2 = generate_lognormal_samples(mu, A, 500, seed=99)
        np.testing.assert_array_equal(X1, X2)
