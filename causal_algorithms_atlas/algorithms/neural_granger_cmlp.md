---
id: neural_granger_cmlp
name: "Neural Granger cMLP"
aliases: ["cMLP", "Neural-GC"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Series approximately stationary over the analyzed window."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "NeuralGrangercMLP"
references: [tank2021]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-09"
---

## Core idea

Fits one neural network (MLP) per target variable, where the first-layer weights are
grouped by source series and penalized with a structured proximal penalty (`GL`,
`GSGL`, or hierarchical) that zeroes out entire weight groups (Tank et al., 2021). A
source series is considered a Granger cause of the target if any weight in its group
remains nonzero after fitting, generalizing the classical Granger test to nonlinear
relationships.

## Assumptions

- **Stationarity: REQUIRED.** The series must be reasonably stationary so the network
  generalizes between the training and evaluation windows.
- **Linearity: NOT required.** This is the method's central motivation: it explicitly
  targets nonlinear lag relationships that classical Granger causality misses.
- The method still assumes that Granger causality (incremental predictive power from
  the past) is the relevant causal question, not structural/interventional causality.

## When to use

When a nonlinear relationship between series is suspected and the predictive (Granger)
interpretation of causality is acceptable in place of a structural one.

## When to avoid

With few observations: per-target neural networks need enough data for the structured
penalty to separate signal from noise, and results tend toward dense output with many
false positives on small samples.

## Relationship to other methods

A direct nonlinear generalization of Classical Granger, following the reference
Neural-GC code from the same authors of the formulation.

## Implementation notes

Wrapper in `causal_discovery/methods/neural_granger.py`. One network per target,
structured proximal penalty configurable via `default_kwargs`.
