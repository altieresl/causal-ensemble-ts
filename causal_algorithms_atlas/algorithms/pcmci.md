---
id: pcmci
name: "PCMCI"
aliases: []
family: constraint-based
temporal_handling: native
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Absence of relevant unmeasured latent confounders."
  - id: stationarity
    required: true
    statement: "Stationary process over the analyzed window."
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
  - id: linearity
    required: true
    statement: "The framework's wrapper uses ParCorr (partial correlation) as the conditional-independence test, which restricts detection to linear Gaussian dependencies."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "PCMCI"
references: [runge2019]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

PCMCI (Runge et al., 2019) combines two stages: first, a condition-selection phase
(PC1) reduces the candidate-parent set of each variable using iterative
conditional-independence tests; then the MCI (Momentary Conditional Independence) test
evaluates each remaining relationship conditioning on both the target's and the
source's estimated parents, which controls for autocorrelation and indirect
confounding at the same time. The framework's wrapper uses `ParCorr` (partial
correlation) as the independence test, so the conditional dependence being tested is
linear even though the general PCMCI formulation is not restricted to that.

## Assumptions

- **Stationarity: REQUIRED.** The process must be stationary over the analyzed
  window.
- **Linearity: REQUIRED (in this framework's configuration).** With `ParCorr`, the
  independence tests additionally assume linear Gaussian relationships for adequate
  statistical power.
- **Causal sufficiency: REQUIRED.** No relevant unmeasured latent confounders.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

Series with many variables and strong temporal autocorrelation, when the absence of
relevant latent confounders can be assumed. A good starting point: it is fast and
returns only definitively oriented lagged relationships (no orientation ambiguity),
which makes interpretation easier.

## When to avoid

When a latent confounder is strongly suspected (consider LPCMCI in that case), or when
the relationship of interest is genuinely nonlinear and the `ParCorr` test masks the
dependency.

## Relationship to other methods

It is the direct predecessor of LPCMCI, which relaxes causal sufficiency at the cost of
returning ambiguous marks instead of fully oriented edges. Compared to Classical
Granger, PCMCI conditions on a data-selected parent set instead of using all available
lags, which reduces false positives in dense networks.

## Implementation notes

Wrapper in `causal_discovery/methods/pcmci.py`, using PCMCI from the `tigramite`
package with `ParCorr`. Returns only definitively oriented lagged relationships.
