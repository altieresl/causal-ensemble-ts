---
id: notears
name: "NOTEARS"
aliases: ["DAGs with NO TEARS"]
family: continuous-optimization
temporal_handling: none
output_type: signed-graph
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Ausencia de confundidores latentes nao medidos."
  - id: acyclicity_instantaneous
    required: true
    statement: "O grafo buscado e aciclico, imposto via restricao continua suave (caracterizacao por traco de exponencial de matriz)."
  - id: linearity
    required: true
    statement: "Formulacao original assume relacoes lineares entre variaveis; extensoes nao lineares existem mas nao sao cobertas por esta ficha."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [zheng2018]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

NOTEARS reformula a busca combinatoria por um DAG (que escala superexponencialmente
com o numero de variaveis) como um problema de otimizacao continua sobre matrizes
reais, usando uma caracterizacao suave e exata de aciclicidade baseada no traco da
exponencial da matriz de adjacencia. Isso permite usar algoritmos numericos padrao em
vez de heuristicas combinatorias de busca de estrutura.

## Premissas

Formulacao original assume relacoes lineares e ausencia de confundidor latente
(suficiencia causal), com aciclicidade garantida explicitamente pela restricao de
otimizacao, nao apenas assumida a priori.

## Quando usar

Como base conceitual antes de considerar DYNOTEARS: util para dados i.i.d.
(nao-temporais) ou como comparacao teorica ao explicar de onde vem a restricao de
aciclicidade continua usada em DYNOTEARS.

## Quando evitar

Para series temporais diretamente: NOTEARS original nao modela lags, apenas a
estrutura instantanea entre variaveis observadas em um unico instante (ou linhas i.i.d.
de um dataset tabular). DYNOTEARS e a extensao que trata explicitamente a dimensao
temporal.

## Relação com outros métodos

E o ancestral direto e nao temporal de DYNOTEARS, ja implementado no framework: DYNOTEARS
adiciona a matriz de coeficientes lagged a mesma formulacao de otimizacao continua com
restricao suave de aciclicidade sobre a parte instantanea.
