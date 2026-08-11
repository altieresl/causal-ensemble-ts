# causal-ensemble-ts

Framework em Python para descoberta causal em séries temporais usando múltiplos
algoritmos, seleção robusta de ensemble, resumo probabilístico, conhecimento
especialista e visualização interativa.

## Visão geral

O projeto representa uma relação temporal pelo contrato:

```text
source(t-lag) -> target(t)
```

Os resultados devem ser interpretados como evidências e hipóteses causais sob as
premissas de cada método, e não como prova definitiva de causalidade em dados
observacionais.

## Pipeline atual

O notebook `Series_Temporais.ipynb` usa somente a pipeline robusta. O antigo fluxo de
ensemble simples não é executado.

Os oito algoritmos candidatos são:

- PCMCI;
- LPCMCI;
- Classical Granger;
- Neural Granger cMLP;
- VAR-LiNGAM;
- DYNOTEARS;
- GES;
- FCI.

Todos são executados na amostra original e nos bootstraps. A pipeline reutiliza essas
saídas para avaliar subconjuntos de métodos sem repetir os ajustes pesados para cada
combinação.

### Adicionando um algoritmo

Os algoritmos são descobertos automaticamente nos módulos de
`causal_discovery/methods/`. Para adicionar um candidato, basta criar uma função
decorada que receba `data` e `max_lag`:

```python
from causal_discovery import causal_method
from causal_discovery.types import canonical_links_to_dataframe


@causal_method()  # run_novo_metodo passa a aparecer como NovoMetodo
def run_novo_metodo(data, max_lag, *, threshold=0.1):
    records = []
    # Ajuste do algoritmo e preenchimento de records.
    return canonical_links_to_dataframe(records)
```

O decorador também aceita `name`, `signed_score`, `weight` e `default_kwargs` quando o
método precisar declarar esses metadados. Não é necessário editar o registro do notebook
ou o `__init__.py`. A execução valida automaticamente o tipo de retorno, as colunas
canônicas, nomes de variáveis, lags, scores e p-valores; uma violação produz
`MethodOutputValidationError` identificando o método e o problema. Depois da inclusão,
o único registro adicional necessário é atualizar a documentação com o novo algoritmo.

Fluxo principal:

1. carregar e pré-processar as séries;
2. configurar os oito métodos;
3. definir o objetivo e as relações de interesse;
4. executar os métodos e filtrar suas saídas para as relações selecionadas;
5. avaliar combinações de métodos com ensemble probabilístico e bootstrap em blocos;
6. escolher a combinação com melhor desempenho agregado;
7. apresentar probabilidades, sinais, estabilidade, consistência e regras especialistas.

## Entrada de dados: CSV e CausalTime

O notebook centraliza a origem em `DATASET_PROFILES` e alterna o perfil por
`ACTIVE_DATASET`. O carregador público `load_time_series_dataset` aceita:

- CSV, com coluna temporal opcional e seleção automática das colunas numéricas;
- CausalTime em `gen_data.npy`, acompanhado por `graph.npy`.

`selected_columns=None` usa dinamicamente todas as variáveis disponíveis. Uma lista
explícita usa somente os nomes informados e é validada contra `available_columns`. A
interface continua derivando seus nós de `processed_data.columns`, portanto passa a
refletir automaticamente o subconjunto carregado.

No CausalTime, nomes ausentes no pacote são gerados como `traffic_00`, `traffic_01` etc.
Os primeiros `N` canais são as variáveis observadas, onde `N` é a dimensão de
`graph.npy`; os canais restantes são auxiliares do gerador e não entram na análise. O
objeto carregado também expõe `ground_truth`, `metadata` e `trajectory_frame(i)`.

As trajetórias do CausalTime são independentes. A descoberta principal escolhe uma delas
por `trajectory_index`. A validação complementar usa `observed_trajectories()` e
`run_pcmci_multiple_trajectories` para analisar todas como datasets separados, sem criar
transições temporais entre o final de uma trajetória e o início da próxima. Cada
variável é padronizada dentro de sua própria trajetória para que diferenças de escala
entre amostras não dominem a dependência temporal. Todos os nós observados entram como
contexto do ajuste; as relações escolhidas na interface são filtradas somente depois.
O perfil atual usa seis nós conectados e `max_lag=2` como teste de integração. Essa
seleção orientada pelo grafo é adequada para depuração, mas não deve ser apresentada como
avaliação cega do benchmark. O grafo de tráfego baixado é simétrico e não contém lags;
logo, sua avaliação estrutural usa `compute_undirected_skeleton_metrics`, que reduz
direções e lags ao mesmo par de nós. A célula posterior à pipeline compara precision,
recall, F1 e SHD com o ground truth, mostra a prevalência de arestas e confronta o F1 do
ensemble com o baseline que prevê todos os pares. Se a interface restringir as relações,
somente os pares efetivamente analisados entram no ground truth da comparação.

O CausalTime original reporta AUROC e AUPRC sobre escores de aresta. Por isso, a mesma
célula calcula também `compute_ranked_undirected_skeleton_metrics` sobre a tabela completa
de pares do PCMCI multi-trajetória. Essa avaliação não depende do limiar binário de 0,5.
No subgrafo atual, a execução reproduzível obteve AUROC `1,000` e average precision
`1,000`; como existem 14 positivos em 15 pares, o baseline aleatório de AP já é `0,933`.
Isso é evidência complementar de ranking, não garantia de zero falsos positivos ou
falsos negativos. O ground truth nunca altera as arestas produzidas pelo ensemble.

## Como as combinações são definidas

Uma combinação é um subconjunto sem repetição dos oito métodos candidatos. A ordem não
importa: `PCMCI + GES` é a mesma combinação que `GES + PCMCI`.

| Execução | Métodos disponíveis | Combinações | Bootstraps padrão | Execuções pré-calculadas |
| --- | ---: | ---: | ---: | ---: |
| Quick | 8 | 28 pares | 4 | `8 x (4 + 1) = 40` |
| Completa | 8 | 28 pares + 56 trios = 84 | 8 | `8 x (8 + 1) = 72` |

Na pipeline principal, `min_votes=2`. Portanto, uma aresta precisa do apoio dos dois
métodos em uma combinação de dois, ou de pelo menos dois dos três métodos em uma
combinação de três.

Cada combinação recebe o seguinte escore:

```text
performance_score =
    0.35 * mean_stability
  + 0.25 * mean_confidence
  + 0.25 * mean_edge_probability
  + 0.15 * mean_method_agreement
```

O ranking é ordenado por `performance_score`, depois por estabilidade e confiança. O
resumo causal final vem da melhor combinação, não necessariamente dos oito métodos ao
mesmo tempo. Assim, “usar todos os algoritmos” significa que todos são executados e
participam da seleção como candidatos.

O pré-cálculo evita executar novamente os métodos para cada um dos 28 ou 84 subconjuntos.
O limite de tempo do bootstrap pode interromper novas reamostragens, de modo que a
quantidade efetiva pode ser menor em máquinas mais lentas.

## Quick e execução completa

Os dois modos usam os mesmos oito algoritmos e os mesmos pesos. A diferença está no custo
da busca:

- Quick avalia apenas pares, usa 4 bootstraps por padrão e limite de 240 segundos para o
  pré-cálculo dos bootstraps;
- Completo avalia pares e trios, usa 8 bootstraps por padrão e limite de 900 segundos.

Quick é apropriado para exploração. A execução completa cobre mais combinações e mais
reamostragens, mas ainda não elimina as limitações dos dados observacionais.

Por enquanto, todos os métodos possuem peso `1.0`. Essa escolha neutra evita introduzir
uma preferência sem calibração experimental: cada método que apoia uma aresta contribui
igualmente para `weighted_support_ratio`. Pesos diferentes só devem ser adotados depois de
uma validação reproduzível em múltiplos cenários, sementes e regimes de ruído.

## Objetivos e relações dinâmicas

A interface gera suas opções a partir das colunas de `processed_data`. Ao trocar o
dataset e reexecutar as células de preparação, configuração e interface, as novas
variáveis passam a aparecer automaticamente.

Objetivos disponíveis:

- explorar toda a estrutura entre variáveis diferentes;
- investigar as possíveis causas de uma variável;
- investigar os possíveis efeitos de uma variável;
- comparar as duas direções entre duas variáveis;
- selecionar relações direcionais específicas.

Autorrelações (`source == target`) não são oferecidas e não são guardadas no resultado
final. A seleção controla quais arestas permanecem nas saídas usadas pelo ensemble. Os
algoritmos ainda recebem o dataset completo, então restringir relações reduz o escopo da
análise, mas não garante redução proporcional do tempo de ajuste.

## Interface e execução padrão

O botão da interface chama `pipeline_runner`. Depois da execução, o resultado fica
armazenado no próprio objeto do dashboard e a célula seguinte o reutiliza.

Se o notebook for executado com “Run All” sem clicar no botão, a célula de resultados
aciona automaticamente uma execução padrão com:

- modo completo (`quick_mode=False`);
- todas as relações entre variáveis diferentes;
- nenhuma regra especialista;
- todos os oito métodos.

Isso evita que regras apenas preenchidas na interface afetem uma execução que não foi
explicitamente iniciada pelo usuário.

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

Relações aceitas:

- `strong`: reforça a existência esperada da aresta;
- `weak`: reduz a expectativa da aresta;
- `inverse`: registra efeito esperado negativo;
- `none`: reduz ou veta a aresta.

Restrições `soft` combinam evidência e prior. Uma regra `none + hard` remove a aresta do
resumo filtrado.

## Saída e interpretação dos escores

Os métodos retornam as colunas canônicas:

```text
source, target, lag, score, p_value, method
```

O significado de `score` depende do algoritmo. Em métodos cujo escore é um coeficiente
com sinal, um valor negativo pode sugerir que o aumento da origem está associado à
redução do alvo, condicionado ao modelo e ao lag. Em GES e FCI, por exemplo, o valor pode
funcionar apenas como marcador estrutural. Por isso, escores brutos de métodos diferentes
não devem ser comparados diretamente.

No resumo robusto, use principalmente:

- `edge_probability`: evidência estimada de existência da aresta;
- `positive_votes` e `negative_votes`: suporte por direção do efeito;
- `sign_consensus` e `sign_agreement`: consenso e concordância sobre o sinal;
- `confidence` e `uncertainty`: confiança e incerteza estimadas;
- estabilidade por bootstrap e métodos que apoiaram a relação.

Uma aresta negativa forte continua sendo evidência de existência. O sinal não deve
cancelar automaticamente a probabilidade da aresta.

## Benchmark e robustez ao ruído

Quando o dataset ativo fornece apenas uma matriz de adjacência simétrica e sem lag, a
validação usa o esqueleto não direcionado: previsões do mesmo par em qualquer direção ou
lag contam como uma única adjacência. Essa avaliação mede recuperação estrutural, mas não
valida orientação causal ou atraso temporal. Para ground truths direcionados e com lag,
o benchmark continua usando `compute_structural_metrics`.

O benchmark auxiliar usa os mesmos oito candidatos, mas avalia somente os 56 trios
possíveis, exige consenso de pelo menos 2 dos 3 métodos, executa 20 bootstraps e adota
`selection_probability_threshold=0.6`. O `max_lag` é 2 e o Neural Granger permanece
limitado a 200 iterações para controlar o custo.

Essa restrição a trios é exclusiva do benchmark. A pipeline principal permanece com 28
pares no Quick e 28 pares mais 56 trios no modo Completo.

Essa política evita que um par muito esparso seja favorecido apenas por concordar sobre
poucas arestas. O consenso de 2 entre 3 preserva controle de falsos positivos enquanto
permite que um método deixe de detectar uma relação mais fraca.

As autorrelações são removidas tanto do `ground_truth` quanto do resumo previsto antes do
cálculo de precision, recall, F1-score e Structural Hamming Distance (SHD). No teste de
ruído severo, uma redução de F1 e um aumento de SHD indicam degradação da recuperação
estrutural.

## Referências

- Runge et al. (2019), *Detecting and quantifying causal associations in large nonlinear time series datasets*.
- Runge (2018), *Causal network reconstruction from time series*.
- Granger (1969), *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*.
- Tank et al. (2021), *Neural Granger Causality*.
- Pamfil et al. (2020), *DYNOTEARS*.
- Meinshausen and Bühlmann (2010), *Stability Selection*.
- Pearl (2009), *Causality: Models, Reasoning, and Inference*.
- Spirtes, Glymour and Scheines (2000), *Causation, Prediction, and Search*.
