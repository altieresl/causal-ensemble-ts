# causal-ensemble-ts

Framework em Python para descoberta causal em series temporais usando ensemble de
metodos, resumo probabilistico, conhecimento especialista e visualizacao interativa.

## Visao geral

O projeto combina resultados de diferentes algoritmos de descoberta causal em uma
representacao comum de arestas temporais:

```text
source -> target em um lag especifico
```

O objetivo e apoiar investigacao causal com mais robustez do que um metodo isolado,
mantendo a incerteza visivel e permitindo que conhecimento de dominio seja aplicado de
forma explicita.

## Fluxo principal

1. Carregue os dados.
2. Aplique `CausalPreprocessor` para preparar as series.
3. Execute metodos candidatos, como PCMCI, LPCMCI, VARLiNGAM, DYNOTEARS ou score-based.
4. Combine resultados com `summarize_probabilistic_ensemble` ou
   `select_robust_ensemble_combination`.
5. Aplique regras especialistas com `expert_knowledge`, quando necessario.
6. Analise o grafo, a tabela de arestas e a consistencia entre metodos no dashboard.

## Uso basico

```python
from causal_discovery import summarize_probabilistic_ensemble

summary = summarize_probabilistic_ensemble(
    all_results,
    min_votes=2,
    method_weights=method_weights,
)

summary.head()
```

## Dashboard interativo

```python
from causal_discovery import create_advanced_expert_dashboard

dashboard = create_advanced_expert_dashboard(
    processed_data=processed_data,
    candidate_methods=candidate_methods,
    candidate_method_kwargs=candidate_method_kwargs,
    method_weights=method_weights,
    all_nodes=list(processed_data.columns),
    pipeline_callback=pipeline_runner,
)
```

O dashboard permite ajustar parametros, cadastrar regras especialistas, rodar o pipeline e
visualizar resultados filtraveis.

## Conhecimento especialista

```python
expert_knowledge = [
    {
        "source": "meanpressure",
        "target": "humidity",
        "lag": 0,
        "relation": "none",
        "constraint": "hard",
        "confidence": 0.95,
        "prior_probability": 0.0,
    },
]
```

Relacoes aceitas:

- `strong`: reforca a existencia esperada da aresta.
- `weak`: reduz a expectativa da aresta.
- `inverse`: marca efeito esperado negativo.
- `none`: reduz ou veta a aresta.

Restricoes aceitas:

- `soft`: mistura evidencia empirica e conhecimento especialista.
- `hard`: aplica uma restricao forte; `none + hard` remove a aresta do resumo filtrado.

## Principais colunas de saida

- `edge_probability`: probabilidade estimada da aresta.
- `posterior_probability`: probabilidade posterior aproximada.
- `combined_p_value`: p-value agregado quando disponivel.
- `support_ratio`: fracao de metodos que apoiam a aresta.
- `support_ci_low` / `support_ci_high`: intervalo de Wilson para suporte.
- `uncertainty`: `1 - edge_probability`.
- `expert_adjustment`: regras especialistas aplicadas.
- `expert_effect`: efeito esperado quando informado pelo especialista.

## Referencias

- Runge et al. (2019), _Detecting and quantifying causal associations in large nonlinear time series datasets_.
- Runge (2018), _Causal network reconstruction from time series_.
- Fisher (1932), _Statistical Methods for Research Workers_.
- Benjamini and Hochberg (1995), _Controlling the False Discovery Rate_.
- Kass and Raftery (1995), _Bayes Factors_.
- Sellke, Bayarri and Berger (2001), _Calibration of p Values for Testing Precise Null Hypotheses_.
- Pearl (2009), _Causality: Models, Reasoning, and Inference_.
- Spirtes, Glymour and Scheines (2000), _Causation, Prediction, and Search_.
