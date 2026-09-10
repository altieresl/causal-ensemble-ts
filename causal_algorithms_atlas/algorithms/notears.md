---
id: notears
name: "NOTEARS"
aliases: ["DAGs with NO TEARS"]
family: continuous-optimization
temporal_handling: none
output_type: signed-graph
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Absence of relevant unmeasured latent confounders."
  - id: acyclicity_instantaneous
    required: true
    statement: "The target graph is acyclic, enforced via a smooth continuous constraint (matrix-exponential trace characterization)."
  - id: linearity
    required: true
    statement: "The original formulation assumes linear relationships between variables; nonlinear extensions exist but are not covered by this card."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [zheng2018]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

NOTEARS (Zheng et al., 2018) reformulates the combinatorial search for a DAG (which
scales super-exponentially with the number of variables) as a continuous optimization
problem over real-valued matrices, using a smooth, exact acyclicity characterization
based on the trace of the matrix exponential of the adjacency matrix. This allows
standard numerical solvers to replace combinatorial structure-search heuristics.

## Assumptions

- **Stationarity: not applicable.** NOTEARS is not natively temporal (see "When to
  avoid").
- **Linearity: REQUIRED (original formulation).** Nonlinear extensions exist but are
  not covered by this card.
- **Acyclicity (instantaneous): REQUIRED.** Guaranteed explicitly by the optimization
  constraint, not merely assumed a priori.
- **Causal sufficiency: REQUIRED.** Absence of relevant unmeasured latent confounders.

## When to use

As a conceptual foundation before considering DYNOTEARS: useful for i.i.d.
(non-temporal) data, or as a theoretical comparison when explaining where DYNOTEARS'
continuous acyclicity constraint comes from.

## When to avoid

Directly for time series: the original NOTEARS does not model lags, only the
instantaneous structure between variables observed at a single instant (or i.i.d. rows
of a tabular dataset). DYNOTEARS is the extension that explicitly handles the temporal
dimension.

## Relationship to other methods

It is the direct, non-temporal ancestor of DYNOTEARS, already implemented in the
framework: DYNOTEARS adds the lagged coefficient matrix to the same continuous
optimization formulation with a smooth acyclicity constraint on the instantaneous part.
