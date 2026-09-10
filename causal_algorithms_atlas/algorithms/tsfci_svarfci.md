---
id: tsfci_svarfci
name: "tsFCI / SVAR-FCI"
aliases: ["time series FCI", "Structural VAR FCI"]
family: constraint-based
temporal_handling: native
output_type: pag
assumptions:
  - id: stationarity
    required: true
    statement: "Causally stationary process: the causal relationship between variables at different time points does not change over time."
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [entner2010, malinsky2018]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

tsFCI (Entner & Hoyer, 2010) adapts FCI to causally stationary time series, by default
orienting lagged edges with a circle mark on one end (valid because an effect cannot
precede its cause), while allowing latent confounding between lagged and contemporaneous
variables. SVAR-FCI (Malinsky & Spirtes, 2018) is a later adaptation that uses the
stationarity assumption more aggressively, automatically removing additional edges
between lagged and contemporaneous variables across all time points whenever the
corresponding independence is detected, producing a more informative PAG than tsFCI
under the same assumptions.

## Assumptions

- **Stationarity: REQUIRED (causal stationarity).**
- **Linearity: NOT required by the general formulation** (depends on the
  conditional-independence test chosen in practice).
- **Causal sufficiency: NOT required.** This is the central motivation of both methods
  relative to a temporal adaptation of PC.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

When a latent confounder is suspected in a time series and a partially oriented output
(PAG) is acceptable; SVAR-FCI is preferable to tsFCI when full stationarity is a
defensible assumption, since it produces a more informative output under that stronger
assumption.

## When to avoid

When full causal stationarity is implausible (a regime change in the causal
relationship over time) -- SVAR-FCI in particular depends on this assumption for its
additional edge removals.

## Relationship to other methods

They occupy the same niche as LPCMCI (tolerating latent confounders, PAG output), but
within the classical FCI/PC family rather than Tigramite's MCI test; it is the direct
temporal analogue of FCI, the same way PCMCI+ and LPCMCI are natively temporal
extensions of the PC/MCI family.
