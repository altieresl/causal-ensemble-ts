---
id: timino
name: "TiMINo"
aliases: ["Time Series Models with Independent Noise"]
family: functional-causal-model
temporal_handling: native
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: false
    statement: "Nao exige suficiencia causal nem fidelidade — o metodo e identificavel mesmo sob nao-fidelidade, segundo o paper original."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [peters2013]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

TiMINo modela cada serie como uma equacao estrutural (SEM) restrita em funcao de seus
pais causais (lagged e/ou instantaneos) mais um ruido independente. A restricao sobre
a classe de funcoes admissiveis e o que garante identificabilidade: ao contrario de
Granger causality, que explora variancia dos residuos, TiMINo explora a independencia
estatistica entre residuo e causas — testando qual atribuicao causal produz residuos
efetivamente independentes das variaveis de entrada.

## Premissas

Segundo o paper original, TiMINo nao exige suficiencia causal nem fidelidade causal, e
tolera realimentacao nao instantanea entre as series. A identificabilidade depende da
classe restrita de funcoes assumida para as equacoes estruturais (nao lineares, mas
nao arbitrarias).

## Quando usar

Quando se aceita a estrutura de modelo de equacoes estruturais com ruido independente e
ha suspeita de relacoes nao lineares, lagged ou instantaneas, incluindo possivel
realimentacao nao instantanea entre variaveis.

## Quando evitar

Quando a classe de funcoes estruturais assumida pelo TiMINo nao for uma aproximacao
razoavel do processo gerador real — como em todo metodo baseado em modelo funcional
restrito, a identificabilidade depende dessa escolha.

## Relação com outros métodos

Ocupa a mesma familia (functional-causal-model) que VAR-LiNGAM no framework, mas troca
a identificacao via nao-gaussianidade linear do LiNGAM por independencia entre residuo
e causas sob uma classe de funcoes nao lineares mais geral.
