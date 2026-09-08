---
id: pcmci_plus
name: "PCMCI+"
aliases: []
family: constraint-based
temporal_handling: native
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Ausencia de confundidores latentes nao medidos."
  - id: stationarity
    required: true
    statement: "Processo estacionario no intervalo analisado."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [runge2020pcmciplus]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

PCMCI+ estende PCMCI para descobrir tambem relacoes contemporaneas (lag 0), nao apenas
defasadas. Usa um esquema de selecao de condicionantes separado para o grafo lagged e
para o esqueleto contemporaneo, o que evita que a forte autocorrelacao tipica de series
temporais infle falsos positivos nas relacoes instantaneas — problema que o paper
mostra afetar PC e outros metodos aplicados ingenuamente ao caso temporal. A formulacao
geral do teste de independencia condicional nao exige linearidade, embora a escolha
pratica do teste (ex.: `ParCorr`) determine se a instancia usada e linear.

## Premissas

Suficiencia causal, estacionariedade e fidelidade causal, como PCMCI. Diferente de
LPCMCI, ainda assume ausencia de confundidores latentes.

## Quando usar

Quando ha razao para esperar efeitos contemporaneos relevantes (lag 0) alem dos
defasados e nao ha suspeita forte de confundidor latente — cobre uma lacuna que o
PCMCI original (apenas lagged) deixa e que o framework atual preenche parcialmente
apenas com VAR-LiNGAM e DYNOTEARS (ambos lineares).

## Quando evitar

Quando ha suspeita de confundidor latente: PCMCI+ mantem a suposicao de suficiencia
causal, diferente de LPCMCI.

## Relação com outros métodos

E o proximo passo natural apos PCMCI antes de introduzir a tolerancia a confundidor
latente do LPCMCI; cobre o mesmo tipo de lacuna que VAR-LiNGAM e DYNOTEARS cobrem no
framework atual (efeitos contemporaneos), mas dentro do paradigma constraint-based em
vez de functional-causal-model/continuous-optimization.
