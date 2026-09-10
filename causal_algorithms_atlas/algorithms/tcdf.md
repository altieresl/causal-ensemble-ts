---
id: tcdf
name: "TCDF"
aliases: ["Temporal Causal Discovery Framework"]
family: functional-causal-model
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: causal_sufficiency
    required: false
    statement: "The paper reports that the method can, in certain circumstances, indicate the presence of hidden confounders, but does not treat this as a general guarantee of the method."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [nauta2019]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

TCDF (Nauta et al., 2019) trains a temporal convolutional network with an attention
mechanism per target variable; the learned attention weights indicate which source
series are relevant for predicting the target, and a separate causal-validation step
(based on intervening on the input data) filters out spurious relationships before
reporting the final causal graph. By interpreting the network's internal parameters,
TCDF also estimates the time delay between cause and effect.

## Assumptions

- **Stationarity: not declared as a formal precondition.**
- **Linearity: NOT required.** The convolutional network captures nonlinear patterns
  by design.
- **Causal sufficiency: NOT required as a guarantee.** The paper reports that the
  method can, in certain circumstances, indirectly signal the presence of hidden
  confounders, but this is not a general guarantee of the method. Results depend on the
  quality of the causal-validation step to distinguish predictive attention from
  genuine causal relationships.

## When to use

When there is enough data to train a per-target network and nonlinear relationships
with an unknown time delay are suspected -- the method estimates that delay itself as a
byproduct.

## When to avoid

With few observations, for the same reason that limits Neural Granger cMLP:
per-target networks need enough data for the causal-validation step to distinguish real
signal from spurious correlation captured by attention.

## Relationship to other methods

Occupies a space close to Neural Granger cMLP in the framework (neural-network-based
approach for nonlinear per-target relationships), but uses attention in a temporal CNN
plus an explicit causal-validation step instead of a structured proximal penalty over
the input weights.
