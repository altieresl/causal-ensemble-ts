---
id: ccm
name: "Convergent Cross Mapping"
aliases: ["CCM"]
family: state-space-reconstruction
temporal_handling: native
output_type: unsigned-graph
assumptions:
  - id: deterministic_dynamics
    required: true
    statement: "The system follows low-dimensional deterministic dynamics, reconstructible via Takens embedding (not dominated by stochastic noise)."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [sugihara2012]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

CCM (Sugihara et al., 2012) is grounded in dynamical-systems theory (Takens embedding):
if `X` causes `Y` in a coupled dynamical system, information about `X`'s state is
imprinted on `Y`'s reconstructed attractor, allowing values of `X` to be estimated from
neighbors on `Y`'s attractor. Causality is inferred when this "cross mapping" ability
improves (converges) as more data is used to reconstruct the attractor.

## Assumptions

- **Stationarity: not required in the classical sense** -- the method targets
  deterministic dynamical systems rather than stochastic stationary processes.
- **Linearity: NOT required.** It targets nonlinear dynamics by design.
- **Deterministic dynamics: REQUIRED.** The system must follow low-dimensional
  deterministic dynamics, the opposite regime from what classical Granger causality was
  designed for (linearly separable stochastic systems). It does not assume the absence
  of a latent confounder in the way `causal_sufficiency` describes, but it does depend
  on the system being a genuinely coupled dynamical system, not noise-dominated.

## When to use

Weakly to moderately coupled systems where deterministic nonlinear dynamics are
suspected (e.g., ecology, physiological systems) and where classical Granger
causality's separability assumption is implausible.

## When to avoid

Systems dominated by stochastic noise, or with very strong/synchronized coupling (in
that extreme regime, bidirectional cross mapping may not distinguish the causal
direction well). Also unsuitable when the domain does not suggest a low-dimensional
dynamical system.

## Relationship to other methods

It is conceptually distinct from all eight methods implemented in the framework: none
of them rely on state-space reconstruction (Takens embedding). It is the most commonly
cited alternative when Granger-causality assumptions (including Neural Granger cMLP)
are considered inadequate for strongly nonlinear, deterministic dynamical systems.
