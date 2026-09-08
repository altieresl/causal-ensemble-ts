---
id: pcmci
name: "PCMCI"
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
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "PCMCI"
references: [runge2019]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

PCMCI combina duas etapas: primeiro, uma selecao de condicionantes (PC1) reduz o
conjunto de pais candidatos de cada variavel usando testes de independencia condicional
iterativos; depois, o teste MCI (Momentary Conditional Independence) avalia cada
relacao remanescente condicionando tanto nos pais estimados do alvo quanto nos da
origem, o que controla autocorrelacao e confundimento indireto ao mesmo tempo. O
wrapper do framework usa `ParCorr` (correlacao parcial) como teste de independencia,
portanto a dependencia condicional testada e linear mesmo que o metodo em si nao seja
restrito a isso na formulacao geral.

## Premissas

Exige suficiencia causal (nenhum confundidor latente relevante), estacionariedade no
trecho analisado e fidelidade causal. Com `ParCorr`, adicionalmente assume relacoes
lineares gaussianas para os testes de independencia terem poder estatistico adequado.

## Quando usar

Series com muitas variaveis e autocorrelacao temporal forte, quando se pode assumir
ausencia de confundidores latentes relevantes. Bom ponto de partida por ser rapido e
por retornar apenas relacoes lagged definitivamente direcionadas (sem ambiguidade de
orientacao), o que facilita interpretacao.

## Quando evitar

Quando ha suspeita forte de confundidor latente (nesse caso, considerar LPCMCI) ou
quando a relacao de interesse e genuinamente nao linear e o teste `ParCorr` mascara a
dependencia.

## Relação com outros métodos

E o predecessor direto do LPCMCI, que relaxa a suficiencia causal ao custo de retornar
marcas ambiguas em vez de arestas totalmente orientadas. Comparado a Classical Granger,
PCMCI condiciona em um conjunto de pais selecionado por dados em vez de usar todos os
lags disponiveis, o que reduz falsos positivos em redes densas.

## Notas de implementação

Wrapper em `causal_discovery/methods/pcmci.py`, usando o PCMCI do pacote `tigramite`
com `ParCorr`. Retorna apenas relacoes lagged definitivamente direcionadas.
