---
id: dynotears
name: "DYNOTEARS"
aliases: []
family: continuous-optimization
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Linear lagged and instantaneous relationships."
  - id: acyclicity_instantaneous
    required: true
    statement: "The instantaneous (lag 0) structure is acyclic, enforced via a smooth continuous constraint."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: both
implemented_in_framework: true
framework_method_name: "DYNOTEARS"
references: [pamfil2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

Formulates causal discovery as a continuous optimization problem (Pamfil et al.,
2020): it simultaneously learns the lagged and instantaneous coefficient matrices by
minimizing reconstruction error with an L1 (sparsity) penalty, subject to a smooth
(differentiable) acyclicity constraint on the instantaneous structure -- NOTEARS
extended to the temporal case.

## Assumptions

- **Stationarity: NOT required.** The method does not declare stationarity as a
  precondition; the optimization formulation accepts either a single long series or a
  panel of short replicates.
- **Linearity: REQUIRED.** All relationships (lagged and instantaneous) are assumed
  linear.
- **Acyclicity (instantaneous): REQUIRED.** Enforced explicitly by the optimization
  constraint -- not merely assumed -- the method fails to converge to a structure with
  instantaneous cycles by construction.

## When to use

When the number of variables is moderate to large and the linear formulation is
acceptable; also useful when the data is a panel of series (multiple short replicates)
rather than a single long series, since the optimization formulation accepts both.

## When to avoid

When the instantaneous-acyclicity constraint is implausible for the domain (e.g., if
an instantaneous feedback loop between variables is expected).

## Relationship to other methods

It is VAR-LiNGAM's counterpart, trading identification via non-Gaussianity for an
explicit optimization constraint to orient the instantaneous structure; it does not
require non-Gaussian residuals.

## Implementation notes

Local implementation in `causal_discovery/methods/dynotears.py`: linear formulation
with a smooth acyclicity constraint and a shared L1 penalty. `signed_score=True`.
