# causal-ensemble-ts

Ensemble Causal para Series Temporais

## Resumo Causal Probabilistico

O projeto agora inclui uma camada probabilistica para representar a incerteza na descoberta causal.

Use summarize_probabilistic_ensemble para sair de uma visao binaria
("a aresta existe" vs "a aresta nao existe") para probabilidades interpretaveis.

### O que essa camada combina

- Evidencia estatistica: p-values combinados com o metodo de Fisher.
- Atualizacao bayesiana de evidencia: probabilidade a posteriori via aproximacao de Bayes Factor.
- Confianca de concordancia: razao de votos e intervalo de confianca de Wilson entre metodos.

### Uso basico

```python
from causal_discovery import summarize_probabilistic_ensemble

summary = summarize_probabilistic_ensemble(all_results, min_votes=2)
print(summary.head())
```

### Principais colunas de saida

- edge_probability: estimativa final de probabilidade para source -> target em um lag.
- posterior_probability: probabilidade bayesiana a partir da evidencia estatistica.
- combined_p_value: p-value agregado quando disponivel.
- support_ratio: fracao de metodos que apoiam a aresta.
- support_ci_low / support_ci_high: intervalo de confianca para a razao de suporte.
- uncertainty: 1 - edge_probability.

## Referencias Para Estudo

### Descoberta causal em series temporais

- Runge et al. (2019) - Detecting and quantifying causal associations in large nonlinear time series datasets.
  Link: https://www.science.org/doi/10.1126/sciadv.aau4996
- Runge (2018) - Causal network reconstruction from time series: From theoretical assumptions to practical estimation.
  Link: https://doi.org/10.1063/1.5025050

### Evidencia estatistica e combinacao de p-values

- Fisher (1932) - Statistical Methods for Research Workers (base do metodo de Fisher para combinacao de testes).
  Link: https://archive.org/details/in.ernet.dli.2015.547535
- Benjamini and Hochberg (1995) - Controlling the False Discovery Rate.
  Link: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

### Interpretacao bayesiana da evidencia

- Kass and Raftery (1995) - Bayes Factors.
  Link: https://doi.org/10.1080/01621459.1995.10476572
- Sellke, Bayarri and Berger (2001) - Calibration of p Values for Testing Precise Null Hypotheses.
  Link: https://doi.org/10.1080/01621459.2001.10475052

### Fundamentos de causalidade

- Pearl (2009) - Causality: Models, Reasoning, and Inference.
  Link: https://www.cambridge.org/highereducation/books/causality/6F5E1A8E6E3D3A2D32D4D1DDAA6E5A9D
- Spirtes, Glymour and Scheines (2000) - Causation, Prediction, and Search.
  Link: https://mitpress.mit.edu/9780262693158/causation-prediction-and-search/
