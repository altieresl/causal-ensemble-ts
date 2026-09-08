---
id: classical_granger
name: "Classical Granger"
aliases: ["Granger Causality"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Series estacionarias (ou tornadas estacionarias por diferenciacao previa)."
  - id: linearity
    required: true
    statement: "Relacao preditiva linear entre os lags e o alvo."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "ClassicalGranger"
references: [granger1969]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Testa se os valores passados de uma serie `X` melhoram a previsao linear de outra
serie `Y` alem do que os proprios valores passados de `Y` ja explicam. E um teste
bivariado e estritamente preditivo: "causa" aqui significa "tem poder preditivo
incremental", nao causalidade estrutural no sentido de intervencao.

## Premissas

Estacionariedade das series e uma forma funcional linear entre lags e alvo. O teste
conjunto usado no wrapper considera todos os lags simultaneamente, mas a significancia
de cada coeficiente individual e reportada separadamente.

## Quando usar

Como baseline rapido e interpretavel, ou quando a relacao realmente e bivariada e
aproximadamente linear. Boa referencia de comparacao para os metodos multivariados do
ensemble.

## Quando evitar

Quando ha confundidores comuns entre `X` e `Y` que nao entram no teste bivariado: nesse
caso, causalidade de Granger bivariada pode indicar uma relacao espuria induzida pelo
confundidor. Tambem inadequado se a relacao for genuinamente nao linear.

## Relação com outros métodos

E o fundamento historico de toda a familia "Granger-based" do projeto, incluindo Neural
Granger cMLP, que generaliza o teste para relacoes nao lineares multivariadas com uma
rede por alvo.

## Notas de implementação

Wrapper em `causal_discovery/methods/classical_granger.py`, usando o teste de
causalidade de Granger do Statsmodels. `signed_score=True` no registro do framework.
