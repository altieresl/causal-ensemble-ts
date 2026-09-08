---
id: dynotears
name: "DYNOTEARS"
aliases: []
family: continuous-optimization
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Relacoes lagged e instantaneas lineares."
  - id: acyclicity_instantaneous
    required: true
    statement: "Estrutura instantanea (lag 0) e aciclica, imposta via restricao continua suave."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: both
implemented_in_framework: true
framework_method_name: "DYNOTEARS"
references: [pamfil2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Formula a descoberta causal como um problema de otimizacao continua: aprende
simultaneamente as matrizes de coeficientes lagged e instantanea minimizando erro de
reconstrucao com penalizacao L1 (esparsidade) sujeita a uma restricao de aciclicidade
suave (diferenciavel) sobre a estrutura instantanea, no estilo NOTEARS estendido para o
caso temporal.

## Premissas

Linearidade de todas as relacoes e aciclicidade da estrutura instantanea, imposta
explicitamente pela restricao de otimizacao (nao apenas assumida — o metodo falha em
convergir para uma estrutura com ciclos instantaneos por construcao).

## Quando usar

Quando o numero de variaveis e moderado a grande e se aceita a formulacao linear;
tambem util quando os dados sao um painel de series (multiplas replicas curtas) em vez
de uma unica serie longa, ja que a formulacao de otimizacao aceita ambos.

## Quando evitar

Quando a restricao de aciclicidade instantanea for implausivel para o dominio (ex.: se
ha razao para esperar um ciclo de feedback instantaneo entre variaveis).

## Relação com outros métodos

E a contraparte de VAR-LiNGAM que troca a identificacao via nao-gaussianidade por uma
restricao explicita de otimizacao para orientar a estrutura instantanea; nao exige
nao-gaussianidade dos residuos.

## Notas de implementação

Implementacao local em `causal_discovery/methods/dynotears.py`: formulacao linear com
restricao suave de aciclicidade e penalizacao L1 compartilhada. `signed_score=True`.
