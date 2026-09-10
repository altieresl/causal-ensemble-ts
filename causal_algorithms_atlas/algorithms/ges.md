---
id: ges
name: "GES"
aliases: ["Greedy Equivalence Search"]
family: score-based
temporal_handling: windowed-unrolling
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Absence of relevant unmeasured latent confounders."
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
  - id: linearity
    required: true
    statement: "The framework's wrapper uses the local_score_BIC score (Gaussian BIC) by default, which assumes linear relationships with Gaussian residuals."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "GES"
references: [chickering2002]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

GES (Chickering, 2002) is a general (non-temporal) algorithm that performs a
two-phase greedy search (forward, adding edges; backward, removing them) over the
space of Markov equivalence classes, optimizing a score (BIC by default) until it
reaches a local optimum that theory guarantees is the correct structure when the
assumptions hold and enough data is available.

## Assumptions

- **Stationarity: NOT required.** No stationarity precondition is declared.
- **Linearity: REQUIRED (in this framework's configuration).** The score used
  (Gaussian BIC) assumes a functional form compatible with linear relationships and
  Gaussian residuals.
- **Causal sufficiency: REQUIRED.** Inherited from the general (non-temporal)
  algorithm.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

As an additional candidate alongside natively temporal methods (PCMCI, LPCMCI), to
capture structure that a score-based search may find and a constraint-based search may
miss, especially with few variables where the greedy search is cheap.

## When to avoid

GES and FCI are general algorithms for tabular data, not designed for time series.
They should not be treated as stationarity-aware temporal algorithms like PCMCI just
because they have been adapted.

## Relationship to other methods

In the framework, GES and FCI receive the same adaptation: a matrix unrolled in time
(`variable_t`, `variable_lag_1`, ...), and the conversion back to the temporal
contract only considers relationships between a lagged variable and a current one.
Unoriented edges between past and present are oriented by the known temporal order.

## Implementation notes

Wrapper in `causal_discovery/methods/causal_learn.py`, using the official GES from the
`causal-learn` package over the expanded temporal matrix.
