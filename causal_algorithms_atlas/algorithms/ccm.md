---
id: ccm
name: "Convergent Cross Mapping"
aliases: ["CCM"]
family: state-space-reconstruction
temporal_handling: native
output_type: unsigned-graph
assumptions:
  - id: deterministic_dynamics
    required: true
    statement: "O sistema segue uma dinamica deterministica de baixa dimensionalidade, reconstruivel via embedding de Takens (nao dominada por ruido estocastico)."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: false
framework_method_name: null
references: [sugihara2012]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Ideia central

CCM parte da teoria de sistemas dinamicos (embedding de Takens): se `X` causa `Y` em
um sistema dinamico acoplado, entao informacao sobre o estado de `X` fica impressa no
atrator reconstruido de `Y`, permitindo estimar valores de `X` a partir de vizinhos no
atrator de `Y`. A causalidade e inferida quando essa capacidade de "cross mapping"
melhora (converge) a medida que mais dados sao usados para reconstruir o atrator.

## Premissas

Assume dinamica deterministica de baixa dimensionalidade — o oposto do regime em que
Granger causality classico foi pensado (sistemas estocasticos linearmente separaveis).
Nao assume ausencia de confundidor latente da forma que causal_sufficiency descreve,
mas depende de o sistema ser genuinamente um sistema dinamico acoplado, nao dominado
por ruido.

## Quando usar

Sistemas fracamente a moderadamente acoplados onde se suspeita de dinamica nao linear
determinista (ex.: ecologia, sistemas fisiologicos) e onde a premissa de separabilidade
do Granger causality classico e implausivel.

## Quando evitar

Sistemas dominados por ruido estocastico ou com acoplamento muito forte/sincronizado
(nesse regime extremo, cross mapping bidirecional pode nao distinguir bem a direcao
causal). Tambem inadequado quando o dominio nao sugere um sistema dinamico de baixa
dimensionalidade.

## Relação com outros métodos

E conceitualmente distinto de todos os oito metodos do framework: nenhum deles se
baseia em reconstrucao de espaco de estados (embedding de Takens). E o metodo mais
citado como alternativa quando as premissas de Granger causality (incluindo Neural
Granger cMLP) sao consideradas inadequadas para sistemas dinamicos fortemente nao
lineares e deterministicos.
