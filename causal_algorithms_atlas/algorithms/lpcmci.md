---
id: lpcmci
name: "LPCMCI"
aliases: []
family: constraint-based
temporal_handling: native
output_type: pag
assumptions:
  - id: stationarity
    required: true
    statement: "Stationary process over the analyzed window."
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
  - id: linearity
    required: true
    statement: "The framework's wrapper uses ParCorr (partial correlation) as the conditional-independence test, which restricts detection to linear Gaussian dependencies."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "LPCMCI"
references: [gerhardus2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

LPCMCI (Gerhardus & Runge, 2020) generalizes PCMCI to the latent-confounder case,
returning a DPAG (Directed Partial Ancestral Graph) instead of a fully oriented DAG.
Edge marks can remain ambiguous (`o-o`, `o->`) when the data does not allow the
orientation to be determined with confidence in the presence of unobserved
confounders.

## Assumptions

- **Stationarity: REQUIRED.** The process must be stationary over the analyzed
  window.
- **Linearity: REQUIRED (in this framework's configuration).** The framework's
  wrapper uses `ParCorr`, inheriting the same restriction to linear dependencies as
  PCMCI.
- **Causal sufficiency: NOT required.** This is the method's central motivation.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

When a relevant latent confounder is reasonably suspected and a partially oriented
output (PAG) is acceptable instead of a complete DAG.

## When to avoid

When a fully directed edge is needed for every candidate relationship: the framework
does not convert ambiguous or bidirected DPAG marks into causal arrows, so relationships
with an ambiguous mark simply do not appear as directed evidence in the output.

## Relationship to other methods

Relaxes PCMCI's strongest assumption (causal sufficiency) at the cost of orientation
ambiguity, analogous to how FCI generalizes PC/GES in the non-temporal case.

## Implementation notes

Wrapper in `causal_discovery/methods/lpcmci.py`, using LPCMCI from `tigramite` with
`ParCorr`. Ambiguous or bidirected marks are not converted into causal edges.
