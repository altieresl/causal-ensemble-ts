---
id: ges
name: "GES"
aliases: ["Greedy Equivalence Search"]
family: score-based
temporal_handling: windowed-unrolling
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Ausencia de confundidores latentes nao medidos."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "GES"
references: [chickering2002]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

GES e um algoritmo geral (nao temporal) que busca greedy em duas fases (forward,
adicionando arestas; backward, removendo) sobre o espaco de classes de equivalencia de
Markov, otimizando um score (por padrao, BIC) ate atingir um otimo local que a teoria
garante ser a estrutura correta quando as premissas valem e os dados sao suficientes.

## Premissas

Suficiencia causal e fidelidade — heranca do algoritmo geral, nao especifico ao caso
temporal. Ao ser adaptado para series no framework, o score usado assume forma
funcional compativel com BIC gaussiano.

## Quando usar

Como candidato adicional ao lado dos metodos temporais nativos (PCMCI, LPCMCI), para
capturar estrutura que uma busca por score pode encontrar e uma busca baseada em
restricoes pode perder, especialmente com poucas variaveis onde a busca greedy e
barata.

## Quando evitar

GES e FCI sao algoritmos gerais para dados tabulares, nao desenhados para series
temporais. Nao devem ser tratados como algoritmos temporalmente estacionarios como
PCMCI apenas por terem sido adaptados.

## Relação com outros métodos

No framework, GES e FCI recebem a mesma adaptacao: uma matriz desenrolada no tempo
(`variavel_t`, `variavel_lag_1`, ...), e a conversao de volta ao contrato temporal
considera apenas relacoes entre uma variavel defasada e uma variavel atual. Arestas nao
orientadas entre passado e presente sao orientadas pela ordem temporal conhecida.

## Notas de implementação

Wrapper em `causal_discovery/methods/causal_learn.py`, usando o GES oficial do pacote
`causal-learn` sobre a matriz temporal expandida.
