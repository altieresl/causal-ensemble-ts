---
id: transfer_entropy
name: "Transfer Entropy"
aliases: []
family: information-theoretic
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: markov_condition
    required: true
    statement: "O processo pode ser bem aproximado por uma cadeia de Markov de ordem finita para estimar as distribuicoes condicionais envolvidas."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [schreiber2000]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

Transfer entropy mede o quanto o passado de uma serie `X` reduz a incerteza sobre o
estado futuro de `Y`, alem do que o proprio passado de `Y` ja reduz — uma
generalizacao nao parametrica e nao linear da nocao de causalidade de Granger baseada
em teoria da informacao (entropia condicional) em vez de erro de previsao linear.

## Premissas

Nao assume forma funcional (linear ou nao) entre as series, mas exige que as
distribuicoes condicionais envolvidas sejam estimaveis a partir dos dados, o que na
pratica assume que o processo e razoavelmente bem descrito por uma cadeia de Markov de
ordem finita e demanda volume de dados suficiente para estimar entropia condicional
com confiabilidade (a estimacao de entropia e notoriamente sensivel ao tamanho da
amostra e a discretizacao/kernel usados).

## Quando usar

Quando a relacao entre series e fortemente nao linear e nenhuma forma funcional
especifica pode ser assumida a priori, e ha dados suficientes para estimar densidades
ou entropias condicionais com confianca.

## Quando evitar

Com poucas observacoes: a estimacao de transfer entropy exige mais dados que testes
parametricos como Granger classico para obter a mesma confiabilidade estatistica, dado
que depende de estimar distribuicoes condicionais em vez de ajustar poucos parametros.

## Relação com outros métodos

Ocupa o mesmo papel conceitual que Neural Granger cMLP no framework (generalizacao nao
linear da nocao de causalidade de Granger), mas por uma rota nao parametrica baseada em
teoria da informacao em vez de uma rede neural com penalizacao estruturada.
