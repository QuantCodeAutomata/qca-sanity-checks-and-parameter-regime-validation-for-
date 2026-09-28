# Lognormal Quantile-Surface Research

Numerical validation of quantile-surface theory for multivariate lognormal portfolio distributions. All experiments are fully synthetic — no historical financial data required.

## Overview

This repository implements nine experiments that validate the paper's claims about lognormal quantile-surface geometry:

| Exp | Description |
|-----|-------------|
| 1 | Sanity checks and parameter-regime validation |
| 2 | Elliptical benchmark: concavity (left) and convexity (right) |
| 3 | Lognormal left-tail barycentric regime: concavity and optimizer convergence |
| 4 | Lognormal left-tail corner regime (S3): convergence to q₀(t)=t |
| 5 | Lognormal right-tail convexity (S3): convergence to q₀(t)=1-t |
| 6 | Surface-measure geometry and mode flow on pulled-back level sets |
| 7 | Tail universality and location-parameter invariance |
| 8 | VaR optimization on the simplex and asymptotic optimizer formulas |
| 9 | Direct validation of quantile gradient and Hessian formulas |

## Repository Structure

```
├── data/               # Placeholder (synthetic data generated in-memory)
├── src/                # Core model and utility implementations
│   ├── model.py        # Scenario setup, SPD checks, lognormal sampling
│   ├── quantile.py     # Empirical quantile (inf-definition), curvature
│   ├── regime.py       # Regime classification, theoretical shapes, Section 5 formulas
│   ├── utils.py        # Simplex grid, normalization, optimizer, error metrics
│   ├── geometry.py     # Level-set pullback, g_y density, mode flow
│   └── integration.py  # Deterministic 2D CDF/quantile inversion, gradient/Hessian validation
├── exp/                # Experiment scripts (exp1–exp9)
├── tests/              # pytest test suite
├── results/            # Output figures and RESULTS.md
└── README.md
```

## Scenarios

| Scenario | mu | B | b | beta | Regime | t* |
|----------|----|---|---|------|--------|----|
| S1 | (0,0) | [[0.5,0.1],[0.1,0.25]] | (0.6, 0.35) | 0.95 | barycentric | 0.6316 |
| S2 | (0,0) | [[5,0.1],[0.1,0.25]] | (5.1, 0.35) | 5.45 | barycentric | 0.9358 |
| S3 | (0,0) | [[0.5,-0.3],[-0.3,0.25]] | (0.2, -0.05) | 0.15 | left_corner | 1.0 |

## Running Experiments

```bash
# Run a single experiment
python exp/exp1_sanity_checks.py

# Run all experiments
for i in 1 2 3 4 5 6 7 8 9; do
    python exp/exp${i}_*.py
done

# Run tests
pytest tests/ -v
```

## Dependencies

- numpy (1.26.4)
- scipy (1.14.1)
- pandas (2.2.2)
- matplotlib

## Key Implementation Notes

- **Empirical quantile**: Uses the inf-definition `ceil(α·N) - 1` with `np.partition` for O(N) speed.
- **Deterministic integration**: `src/integration.py` provides exact 2D CDF inversion for cross-validation.
- **Reproducibility**: All Monte Carlo draws use explicit random seeds.
- **No financial data**: Fully synthetic; no Massive API or external data sources used.
