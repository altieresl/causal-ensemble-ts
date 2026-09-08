---
id: tcdf
name: "TCDF"
aliases: ["Temporal Causal Discovery Framework"]
family: functional-causal-model
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: causal_sufficiency
    required: false
    statement: "O paper reporta que o metodo pode, em certas circunstancias, indicar a presenca de confundidores ocultos, mas nao trata isso como garantia geral do metodo."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [nauta2019]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

TCDF treina uma rede convolucional temporal com mecanismo de atencao por
variavel-alvo; os pesos de atencao aprendidos indicam quais series de origem sao
relevantes para prever o alvo, e um passo de validacao causal separado (baseado em
intervencao nos dados de entrada) filtra relacoes espurias antes de reportar o grafo
causal final. Interpretando os parametros internos da rede, TCDF tambem estima o atraso
temporal entre causa e efeito.

## Premissas

Nao assume forma funcional linear (rede convolucional captura padroes nao lineares),
mas depende da qualidade do passo de validacao causal para distinguir atencao
preditiva de relacao causal genuina. O paper reporta que o metodo pode, em certas
circunstancias, sinalizar indiretamente a presenca de confundidores ocultos, mas isso
nao e uma garantia geral do metodo.

## Quando usar

Quando ha volume de dados suficiente para treinar uma rede por alvo e se suspeita de
relacoes nao lineares com atraso temporal desconhecido a priori — o proprio metodo
estima esse atraso como subproduto.

## Quando evitar

Com poucas observacoes, pelo mesmo motivo que limita Neural Granger cMLP: redes por
alvo precisam de dados suficientes para o passo de validacao causal distinguir sinal
real de correlacao espuria capturada pela atencao.

## Relação com outros métodos

Ocupa um espaco proximo ao de Neural Granger cMLP no framework (abordagem baseada em
rede neural para relacoes nao lineares por alvo), mas usa atencao em uma CNN temporal
mais um passo de validacao causal explicito em vez de penalizacao proximal estruturada
sobre os pesos de entrada.
