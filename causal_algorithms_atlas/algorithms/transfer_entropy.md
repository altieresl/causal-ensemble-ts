---
id: transfer_entropy
name: "Transfer Entropy"
aliases: []
family: information-theoretic
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: markov_condition
    required: true
    statement: "The process can be well approximated by a finite-order Markov chain for estimating the involved conditional distributions."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [schreiber2000]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

Transfer entropy (Schreiber, 2000) measures how much the past of a series `X` reduces
uncertainty about the future state of `Y`, beyond what `Y`'s own past already reduces --
a nonparametric, nonlinear generalization of the Granger-causality notion based on
information theory (conditional entropy) instead of linear forecast error.

## Assumptions

- **Stationarity: not declared as a formal precondition**, though estimating stable
  conditional distributions in practice benefits from a reasonably stationary process.
- **Linearity: NOT required.** It makes no assumption about functional form (linear or
  otherwise) between series.
- **Markov condition: REQUIRED.** The process must be well approximated by a
  finite-order Markov chain so the conditional distributions involved are estimable
  from data, which in practice demands enough data volume for reliable conditional
  entropy estimation (entropy estimation is notoriously sensitive to sample size and to
  the discretization/kernel used).

## When to use

When the relationship between series is strongly nonlinear and no specific functional
form can be assumed a priori, and there is enough data to estimate densities or
conditional entropies with confidence.

## When to avoid

With few observations: estimating transfer entropy requires more data than parametric
tests like classical Granger to reach the same statistical reliability, since it
depends on estimating conditional distributions rather than fitting a few parameters.

## Relationship to other methods

Occupies the same conceptual role as Neural Granger cMLP in the framework (nonlinear
generalization of the Granger-causality notion), but via a nonparametric,
information-theoretic route instead of a neural network with structured penalization.
