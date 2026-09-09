---
id: fci
name: "FCI"
aliases: ["Heterogeneous FCI"]
family: constraint-based
temporal_handling: windowed-unrolling
output_type: pag
assumptions:
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
  - id: linearity
    required: true
    statement: "O wrapper do framework usa 'fisherz' como teste de independencia por padrao, o que restringe a deteccao a dependencias lineares gaussianas."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "FCI"
references: [spirtes1995]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

FCI (Fast Causal Inference) generaliza PC para o caso com confundidores latentes e
selecao amostral, retornando um PAG (Partial Ancestral Graph) em vez de um DAG. Como
GES, e um algoritmo geral adaptado ao caso temporal no framework via desenrolamento em
uma matriz `variavel_t`, `variavel_lag_1`, ....

## Premissas

Fidelidade causal — nao exige suficiencia causal, o que e a motivacao central do
metodo frente a PC/GES. No framework, conhecimento previo proibe arestas
presente -> passado na matriz desenrolada, refletindo a ordem temporal conhecida.

## Quando usar

Quando ha suspeita de confundidor latente e se aceita saida parcialmente orientada
(PAG); por padrao o framework retorna apenas arestas PAG definitivamente orientadas,
como o LPCMCI.

## Quando evitar

Mesma ressalva do GES: e um algoritmo geral para dados tabulares, nao nativamente
temporal. A adaptacao usa o algoritmo oficial mas nao o torna estacionario como PCMCI.

## Relação com outros métodos

Papel para GES o mesmo que LPCMCI faz para PCMCI: versão que tolera confundidor latente
trocando DAG totalmente orientado por PAG parcialmente orientado.

## Notas de implementação

Wrapper em `causal_discovery/methods/causal_learn.py` e alias em
`causal_discovery/methods/heterogeneous_fci.py` (`run_heterogeneous_fci` e mantido como
alias compativel de `run_fci`). Usa o FCI oficial do `causal-learn` sobre a matriz
temporal expandida, com conhecimento previo proibindo presente -> passado. Por padrao
retorna apenas arestas PAG definitivamente orientadas.
