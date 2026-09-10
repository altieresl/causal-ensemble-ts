---
id: var_lingam
name: "VAR-LiNGAM"
aliases: []
family: functional-causal-model
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Linear lagged and instantaneous relationships."
  - id: acyclicity_instantaneous
    required: true
    statement: "The instantaneous (lag 0) structure is acyclic."
  - id: non_gaussian_errors
    required: true
    statement: "Independent, non-Gaussian noise terms."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "VARLiNGAM"
references: [hyvarinen2010]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

Combines a VAR (Vector Autoregression) model for the lagged part with LiNGAM (Linear
Non-Gaussian Acyclic Model) to orient the instantaneous (lag 0) structure (Hyvärinen et
al., 2010). Non-Gaussianity of the residuals is what allows the instantaneous causal
direction to be identified without relying solely on conditional-independence
constraints.

## Assumptions

- **Stationarity: NOT required.** The method does not declare stationarity as a
  precondition.
- **Linearity: REQUIRED.** Both lagged and instantaneous relationships must be linear.
- **Acyclicity (instantaneous): REQUIRED.** The lag-0 structure must be acyclic.
- **Non-Gaussian errors: REQUIRED.** Residuals must be independent and non-Gaussian --
  if they were Gaussian, the instantaneous orientation would no longer be identifiable
  under the method's theory.

## When to use

When plausible instantaneous (lag 0) effects exist between the series and there is
reason to believe residuals are non-Gaussian (common in financial data and some
physical sensors).

## When to avoid

With near-Gaussian residuals, or when instantaneous cycles are suspected: LiNGAM's
identifiability depends structurally on non-Gaussianity and on acyclicity.

## Relationship to other methods

It is the only method in the ensemble that explicitly models instantaneous (lag 0)
effects in addition to lagged ones, which makes it complementary to PCMCI and
DYNOTEARS, which mostly or exclusively handle lagged relationships.

## Implementation notes

Wrapper in `causal_discovery/methods/var_lingam.py`, using `lingam.VARLiNGAM`,
including both instantaneous and lagged effects. `signed_score=True` in the framework
registry.
