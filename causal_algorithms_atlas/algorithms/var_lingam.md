---
id: var_lingam
name: "VAR-LiNGAM"
aliases: []
family: functional-causal-model
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Relacoes lagged e instantaneas lineares."
  - id: acyclicity_instantaneous
    required: true
    statement: "Estrutura instantanea (lag 0) e aciclica."
  - id: non_gaussian_errors
    required: true
    statement: "Ruidos independentes e nao gaussianos."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "VARLiNGAM"
references: [hyvarinen2010]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Combina um modelo VAR (Vector Autoregression) para a parte lagged com LiNGAM (Linear
Non-Gaussian Acyclic Model) para orientar a estrutura instantanea (lag 0). A
nao-gaussianidade dos residuos e o que permite identificar a direcao causal instantanea
sem depender apenas de restricoes de independencia condicional.

## Premissas

Linearidade em todas as relacoes (lagged e instantaneas), aciclicidade da estrutura
instantanea, e residuos independentes e nao gaussianos — se os residuos forem
gaussianos, a orientacao instantanea deixa de ser identificavel pela teoria do metodo.

## Quando usar

Quando ha efeitos instantaneos (lag 0) plausiveis entre as series e razao para acreditar
que os residuos nao sao gaussianos (comum em dados financeiros e alguns sensores
fisicos).

## Quando evitar

Com residuos proximos de gaussianos ou quando se suspeita de ciclos na estrutura
instantanea: a identificacao de LiNGAM depende estruturalmente da nao-gaussianidade e
da aciclicidade.

## Relação com outros métodos

E o unico metodo do ensemble que modela explicitamente efeitos instantaneos (lag 0)
alem dos lagged, o que o torna complementar a PCMCI e DYNOTEARS, que tratam
majoritariamente ou exclusivamente relacoes lagged.

## Notas de implementação

Wrapper em `causal_discovery/methods/var_lingam.py`, usando `lingam.VARLiNGAM`, incluindo
efeitos instantaneos e lagged. `signed_score=True` no registro do framework.
