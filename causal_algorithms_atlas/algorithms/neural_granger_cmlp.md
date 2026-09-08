---
id: neural_granger_cmlp
name: "Neural Granger cMLP"
aliases: ["cMLP", "Neural-GC"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Series aproximadamente estacionarias no intervalo analisado."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "NeuralGrangercMLP"
references: [tank2021]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Ajusta uma rede neural (MLP) por variavel-alvo, onde os pesos da primeira camada sao
organizados por serie de origem e penalizados com uma penalizacao proximal estruturada
(`GL`, `GSGL` ou hierarquica) que zera grupos inteiros de pesos. Uma serie de origem e
considerada causa de Granger do alvo se algum peso do seu grupo permanece nao nulo apos
o ajuste, generalizando o teste classico de Granger para relacoes nao lineares.

## Premissas

Nao exige linearidade — essa e a motivacao central do metodo — mas ainda assume que a
nocao de causalidade de Granger (poder preditivo incremental a partir do passado) e a
pergunta relevante, e que a serie e razoavelmente estacionaria para a rede generalizar
entre janelas de treino e avaliacao.

## Quando usar

Quando ha suspeita de relacao nao linear entre series e se aceita a interpretacao
preditiva (Granger) de causalidade em vez de uma estrutural.

## Quando evitar

Com poucas observacoes: redes neurais por alvo precisam de dados suficientes para a
penalizacao estruturada distinguir sinal de ruido, e os resultados tendem a saida densa
com muitos falsos positivos em amostras pequenas.

## Relação com outros métodos

Generalizacao nao linear direta do Classical Granger, seguindo o codigo de referencia
Neural-GC dos mesmos autores da formulacao.

## Notas de implementação

Wrapper em `causal_discovery/methods/neural_granger.py`. Uma rede por alvo, penalizacao
proximal estruturada configuravel via `default_kwargs`.
