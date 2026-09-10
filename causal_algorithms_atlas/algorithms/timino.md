---
id: timino
name: "TiMINo"
aliases: ["Time Series Models with Independent Noise"]
family: functional-causal-model
temporal_handling: native
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: false
    statement: "Requires neither causal sufficiency nor faithfulness -- the method is identifiable even under non-faithfulness, per the original paper."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [peters2013]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

TiMINo (Peters et al., 2013) models each series as a restricted structural equation
(SEM) in terms of its causal parents (lagged and/or instantaneous) plus independent
noise. The restriction on the admissible function class is what guarantees
identifiability: unlike Granger causality, which exploits residual variance, TiMINo
exploits statistical independence between the residual and the causes -- testing which
causal assignment produces residuals that are effectively independent of the input
variables.

## Assumptions

- **Stationarity: not declared as a formal precondition.**
- **Linearity: NOT required.** TiMINo targets nonlinear structural equations, though
  within a restricted function class (see below).
- **Causal sufficiency: NOT required.** Per the original paper, TiMINo requires
  neither causal sufficiency nor faithfulness, and tolerates non-instantaneous
  feedback between series. Identifiability depends on the restricted function class
  assumed for the structural equations (nonlinear, but not arbitrary).

## When to use

When the structural-equation-with-independent-noise framework is acceptable and
nonlinear, lagged, or instantaneous relationships are suspected, including possible
non-instantaneous feedback between variables.

## When to avoid

When TiMINo's assumed structural function class is not a reasonable approximation of
the true generating process -- as with any method based on a restricted functional
model, identifiability hinges on that choice.

## Relationship to other methods

Occupies the same family (functional-causal-model) as VAR-LiNGAM in the framework, but
trades LiNGAM's linear non-Gaussianity-based identification for independence between
residual and causes under a more general nonlinear function class.
