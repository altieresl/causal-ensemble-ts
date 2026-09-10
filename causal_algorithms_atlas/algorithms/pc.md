---
id: pc
name: "PC / PC-stable"
aliases: ["Peter-Clark algorithm"]
family: constraint-based
temporal_handling: none
output_type: partial-graph
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Absence of relevant unmeasured latent confounders."
  - id: faithfulness
    required: true
    statement: "Observed independencies reflect the causal structure, not coincidence."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [spirtes1991, colombo2014]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-09"
---

## Core idea

PC (Spirtes & Glymour, 1991) recovers the Markov equivalence class of a causal graph
by iteratively removing edges between pairs of variables that test conditionally
independent given some subset of the remaining variables, starting from empty sets and
growing the conditioning-set size. The result is a CPDAG (a partially oriented graph
representing the whole equivalence class), not a single DAG. PC-stable (Colombo &
Maathuis, 2014) is a modification that removes the dependence on the order in which
variables are presented to the algorithm, making the skeleton result stable under
reordering.

## Assumptions

- **Stationarity: not applicable.** PC has no native notion of time (see "When to
  avoid").
- **Linearity: not required by the general formulation**, though the independence test
  chosen in practice determines whether the tested dependency is linear.
- **Causal sufficiency: REQUIRED.** No relevant unmeasured latent confounders.
- **Faithfulness: REQUIRED.** Observed independencies must reflect the true causal
  structure.

## When to use

As the reference constraint-based algorithm when the strong assumptions (causal
sufficiency) are acceptable and a simpler, computationally cheaper alternative to
PCMCI is needed for data without temporal structure, or as a conceptual baseline when
comparing against FCI (the version that relaxes causal sufficiency).

## When to avoid

For time series, PC has no native time handling: applying it requires the same kind of
temporal-matrix unrolling used in this framework for GES and FCI, which is not
implemented here. The original PC ordering is also sensitive to input variable order --
PC-stable specifically fixes that problem.

## Relationship to other methods

It is the conceptual predecessor of GES (score search instead of independence tests)
and of FCI (which relaxes PC's causal sufficiency). PCMCI uses a condition-selection
logic reminiscent of PC's skeleton phase, but natively adapted to the temporal case
with the MCI test.
