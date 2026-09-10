---
id: classical_granger
name: "Classical Granger"
aliases: ["Granger Causality"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Stationary series (or made stationary via prior differencing)."
  - id: linearity
    required: true
    statement: "Linear predictive relationship between the lags and the target."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "ClassicalGranger"
references: [granger1969]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

Tests whether the past values of a series `X` improve the linear forecast of another
series `Y` beyond what `Y`'s own past already explains (Granger, 1969). It is a
bivariate, strictly predictive test: "causes" here means "has incremental predictive
power," not structural causality in the interventionist sense.

## Assumptions

- **Stationarity: REQUIRED.** The series must be stationary (or made stationary through
  prior differencing) for the underlying VAR framework and its F-test to be valid.
- **Linearity: REQUIRED.** The predictive relationship between lags and target is
  assumed linear; the framework's wrapper runs a joint test over all lags at once, but
  reports the significance of each individual coefficient separately.

## When to use

As a fast, interpretable baseline, or when the relationship really is bivariate and
approximately linear. A good comparison reference for the ensemble's multivariate
methods.

## When to avoid

When there are common confounders between `X` and `Y` that do not enter the bivariate
test: in that case, bivariate Granger causality can flag a spurious relationship
induced by the confounder. Also unsuitable when the relationship is genuinely
nonlinear.

## Relationship to other methods

It is the historical foundation of the whole "Granger-based" family in this project,
including Neural Granger cMLP, which generalizes the test to multivariate nonlinear
relationships with one network per target.

## Implementation notes

Wrapper in `causal_discovery/methods/classical_granger.py`, using Statsmodels' Granger
causality test. `signed_score=True` in the framework registry.
