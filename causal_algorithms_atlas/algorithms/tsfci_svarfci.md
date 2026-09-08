---
id: tsfci_svarfci
name: "tsFCI / SVAR-FCI"
aliases: ["time series FCI", "Structural VAR FCI"]
family: constraint-based
temporal_handling: native
output_type: pag
assumptions:
  - id: stationarity
    required: true
    statement: "Processo causalmente estacionario: a relacao causal entre variaveis em instantes diferentes nao muda ao longo do tempo."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [entner2010, malinsky2018]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

tsFCI (Entner & Hoyer, 2010) adapta o FCI a series temporais causalmente estacionarias,
orientando por padrao as arestas defasadas com um circulo em uma extremidade (valido
porque um efeito nao pode preceder sua causa), mas permitindo confundimento latente
entre variaveis defasadas e contemporaneas. SVAR-FCI (Malinsky & Spirtes, 2018) e uma
adaptacao posterior que usa a suposicao de estacionariedade de forma mais agressiva,
removendo automaticamente arestas adicionais entre variaveis defasadas e contemporaneas
em todos os instantes de tempo sempre que a independencia correspondente e detectada,
produzindo um PAG mais informativo que o de tsFCI sob as mesmas premissas.

## Premissas

Estacionariedade causal e fidelidade — nao exigem suficiencia causal, que e a
motivacao central de ambos frente a uma adaptacao temporal de PC.

## Quando usar

Quando ha suspeita de confundidor latente em uma serie temporal e se aceita saida
parcialmente orientada (PAG); SVAR-FCI e preferivel a tsFCI quando a estacionariedade
completa e uma suposicao defensavel, pois produz uma saida mais informativa sob essa
suposicao mais forte.

## Quando evitar

Quando a estacionariedade causal completa for implausivel (mudanca de regime na
relacao causal ao longo do tempo) — SVAR-FCI em particular depende dessa suposicao para
as remocoes de aresta adicionais.

## Relação com outros métodos

Ocupam o mesmo nicho que LPCMCI (tolerar confundidor latente, saida PAG), mas dentro da
familia FCI/PC classica em vez do teste MCI do Tigramite; e o analogo temporal direto
de FCI, da mesma forma que PCMCI+ e LPCMCI sao extensoes temporais nativas da familia
PC/MCI.
