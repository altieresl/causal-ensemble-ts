---
id: lpcmci
name: "LPCMCI"
aliases: []
family: constraint-based
temporal_handling: native
output_type: pag
assumptions:
  - id: stationarity
    required: true
    statement: "Processo estacionario no intervalo analisado."
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
implemented_in_framework: true
framework_method_name: "LPCMCI"
references: [gerhardus2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

LPCMCI generaliza o PCMCI para o caso latente, retornando um DPAG (Directed Partial
Ancestral Graph) em vez de um DAG totalmente orientado. Marcas de aresta podem ficar
ambiguas (`o-o`, `o->`) quando os dados nao permitem determinar a orientacao com
seguranca na presenca de confundidores nao observados.

## Premissas

Nao exige suficiencia causal — essa e a motivacao central do metodo. Ainda exige
estacionariedade e fidelidade causal. O wrapper do framework usa `ParCorr`, herdando a
mesma limitacao a dependencia linear que o PCMCI.

## Quando usar

Quando ha suspeita razoavel de confundidor latente relevante e se aceita trabalhar com
saida parcialmente orientada (PAG) em vez de um DAG completo.

## Quando evitar

Quando se precisa de uma aresta totalmente direcionada para toda relacao candidata: o
framework nao converte marcas ambiguas ou bidirecionais do DPAG em setas causais, entao
relacoes com marca ambigua simplesmente nao aparecem como evidencia direcionada na
saida.

## Relação com outros métodos

Relaxa a suposicao mais forte do PCMCI (suficiencia causal) ao custo de ambiguidade de
orientacao, de forma analoga a como FCI generaliza PC/GES no caso atemporal.

## Notas de implementação

Wrapper em `causal_discovery/methods/lpcmci.py`, usando o LPCMCI do `tigramite` com
`ParCorr`. Marcas ambiguas ou bidirecionais nao sao convertidas em arestas causais.
