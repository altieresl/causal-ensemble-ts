---
id: fci
name: "FCI"
aliases: ["Heterogeneous FCI"]
family: constraint-based
temporal_handling: windowed-unrolling
output_type: pag
assumptions:
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
  - id: linearity
    required: true
    statement: "The framework's wrapper uses 'fisherz' as the default independence test, which restricts detection to linear Gaussian dependencies."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "FCI"
references: [spirtes1995]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

FCI (Fast Causal Inference, Spirtes et al., 1995) generalizes PC to the case with
latent confounders and selection bias, returning a PAG (Partial Ancestral Graph)
instead of a DAG. Like GES, it is a general algorithm adapted to the temporal case in
this framework via unrolling into a `variable_t`, `variable_lag_1`, ... matrix.

## Assumptions

- **Stationarity: NOT required.** No stationarity precondition is declared.
- **Linearity: REQUIRED (in this framework's configuration).** The framework's wrapper
  uses Fisher-Z as the default independence test, which restricts detection to linear
  Gaussian dependencies -- even though FCI's general formulation is not tied to a
  specific test.
- **Causal sufficiency: NOT required.** This is FCI's central motivation relative to
  PC/GES -- it explicitly tolerates latent confounders.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure. In the framework, prior knowledge forbids present -> past edges in the
  unrolled matrix, reflecting the known temporal order.

## When to use

When a latent confounder is suspected and a partially oriented output (PAG) is
acceptable; by default the framework returns only definitively oriented PAG edges,
like LPCMCI.

## When to avoid

Same caveat as GES: it is a general algorithm for tabular data, not natively temporal.
The adaptation uses the official algorithm but does not make it stationarity-aware like
PCMCI.

## Relationship to other methods

Plays the same role for GES that LPCMCI plays for PCMCI: a version that tolerates
latent confounders by trading a fully oriented DAG for a partially oriented PAG.

## Implementation notes

Wrapper in `causal_discovery/methods/causal_learn.py`, with an alias in
`causal_discovery/methods/heterogeneous_fci.py` (`run_heterogeneous_fci` is kept as a
compatible alias for `run_fci`). Uses the official `causal-learn` FCI implementation
over the expanded temporal matrix, with prior knowledge forbidding present -> past
edges. By default returns only definitively oriented PAG edges.
