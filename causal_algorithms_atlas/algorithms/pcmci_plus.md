---
id: pcmci_plus
name: "PCMCI+"
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
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [runge2020pcmciplus]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

PCMCI+ (Runge, 2020) extends PCMCI to also discover contemporaneous (lag 0)
relationships, not only lagged ones. It uses a separate condition-selection scheme for
the lagged graph and for the contemporaneous skeleton, which prevents the strong
autocorrelation typical of time series from inflating false positives in the
instantaneous relationships -- a problem the paper shows affects PC and other methods
applied naively to the temporal case. The general conditional-independence test
formulation does not require linearity, though the practical choice of test (e.g.
`ParCorr`) determines whether the instance used is linear.

## Assumptions

- **Stationarity: REQUIRED.** Stationary process over the analyzed window.
- **Linearity: not required by the general formulation** (depends on the
  conditional-independence test chosen in practice).
- **Causal sufficiency: REQUIRED.** Unlike LPCMCI, it still assumes the absence of
  latent confounders.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

When relevant contemporaneous (lag 0) effects are expected in addition to lagged ones,
and there is no strong suspicion of a latent confounder -- it covers a gap that
original PCMCI (lagged only) leaves, which the current framework fills only partially,
with VAR-LiNGAM and DYNOTEARS (both linear).

## When to avoid

When a latent confounder is suspected: PCMCI+ keeps the causal-sufficiency assumption,
unlike LPCMCI.

## Relationship to other methods

It is the natural next step after PCMCI before introducing LPCMCI's tolerance for
latent confounders; it covers the same kind of gap that VAR-LiNGAM and DYNOTEARS cover
in the current framework (contemporaneous effects), but within the constraint-based
paradigm rather than functional-causal-model/continuous-optimization.
