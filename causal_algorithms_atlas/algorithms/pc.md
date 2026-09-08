---
id: pc
name: "PC / PC-stable"
aliases: ["Peter-Clark algorithm"]
family: constraint-based
temporal_handling: none
output_type: partial-graph
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
implemented_in_framework: false
framework_method_name: null
references: [spirtes1991, colombo2014]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

PC (Spirtes & Glymour, 1991) recupera a classe de equivalencia de Markov de um grafo
causal removendo iterativamente arestas entre pares de variaveis que se mostram
condicionalmente independentes dado algum subconjunto das demais variaveis, comecando
por conjuntos vazios e crescendo o tamanho do conjunto condicionante. O resultado e um
CPDAG (grafo parcialmente orientado que representa toda a classe de equivalencia), nao
um DAG unico. PC-stable (Colombo & Maathuis, 2014) e uma modificacao que remove a
dependencia da ordem em que as variaveis sao apresentadas ao algoritmo, tornando o
resultado do skeleton estavel sob reordenacao.

## Premissas

Suficiencia causal (nenhum confundidor latente relevante) e fidelidade causal — as
mesmas premissas fortes de GES. E um algoritmo geral para dados tabulares/i.i.d., sem
nocao nativa de tempo.

## Quando usar

Como algoritmo de referencia da familia constraint-based quando se aceita as premissas
fortes (suficiencia causal) e se precisa de uma alternativa mais simples e mais barata
computacionalmente que PCMCI para dados sem estrutura temporal, ou como baseline
conceitual ao comparar com FCI (a versao que relaxa suficiencia causal).

## Quando evitar

Para series temporais, PC nao tem tratamento nativo do tempo: aplicá-lo exige o mesmo
tipo de desenrolamento em matriz temporal usado no framework para GES e FCI, o que nao
esta implementado aqui. A ordem original de PC tambem e sensivel a ordem das variaveis
de entrada — PC-stable resolve especificamente esse problema.

## Relação com outros métodos

E o predecessor conceitual de GES (busca por score em vez de testes de independencia)
e de FCI (que relaxa a suficiencia causal de PC). PCMCI usa uma logica de selecao de
condicionantes que lembra a fase de skeleton de PC, mas adaptada nativamente ao caso
temporal com o teste MCI.
