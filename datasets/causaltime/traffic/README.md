# Dataset CausalTime Traffic

Esta pasta contém o subconjunto **Traffic** do benchmark CausalTime. Ele foi criado a
partir de séries temporais de sensores de tráfego da região da Baía de São Francisco,
com um grafo de referência derivado da proximidade geográfica entre os sensores.

O conjunto é útil para avaliar descoberta causal porque acompanha um grafo conhecido
pelo processo gerador. Esse grafo é uma referência do benchmark sintético: ele não deve
ser interpretado como prova de causalidade entre sensores no trânsito real.

## Arquivos

### `gen_data.npy`

Array NumPy `float32` com shape `(480, 40, 40)`:

| Dimensão | Significado |
|---|---|
| `480` | Trajetórias independentes geradas pelo benchmark |
| `40` | Passos temporais em cada trajetória |
| `40` | 20 variáveis observadas seguidas de 20 canais residuais/auxiliares |

Os primeiros 20 canais correspondem aos nós observados. Os outros 20 fazem parte do
processo de geração do CausalTime e não são usados como variáveis causais pela pipeline.

As trajetórias são independentes e não devem ser simplesmente concatenadas, pois isso
criaria uma transição temporal artificial entre o fim de uma trajetória e o início da
seguinte.

### `graph.npy`

Matriz de adjacência com shape `(20, 20)`. Ela descreve as conexões de referência entre
os 20 nós observados. Neste conjunto:

- o grafo é simétrico;
- a diagonal representa autorrelações, que são removidas do ground truth usado pelo
  projeto;
- a matriz não informa o atraso (`lag`) exato das relações;
- ao selecionar apenas algumas colunas, o carregador conserva somente as relações entre
  os nós selecionados.

Uma entrada ligando os nós 2 e 4 significa que o benchmark permite uma relação entre
essas séries em algum atraso histórico. Ela não identifica o local físico dos sensores
nem demonstra uma relação causal real entre duas vias específicas.

## Significado das colunas

Os arquivos `.npy` não armazenam nomes de colunas, identificadores dos sensores,
coordenadas ou nomes de rodovias. Por isso, o carregador do projeto gera nomes com o
prefixo configurado:

```text
traffic_00, traffic_01, ..., traffic_19
```

Cada coluna representa o sinal temporal de **um sensor/nó diferente**. Elas não
representam medições distintas como velocidade, fluxo e ocupação. Por exemplo:

- `traffic_02`: sinal do nó de índice 2;
- `traffic_13`: sinal do nó de índice 13;
- `traffic_19`: sinal do nó de índice 19.

O benchmark tem origem em dados de sensores de tráfego, mas o pacote gerado não preserva
metadados suficientes para converter seus valores com segurança para km/h, mph ou outra
unidade física. Os valores devem ser interpretados como sinais gerados/escalados, úteis
principalmente para comparação estrutural e relativa.

O índice `time` produzido pelo carregador vai de `0` a `39` e identifica os passos da
trajetória. Ele não representa datas ou horários reais preservados no arquivo.

## Como carregar no projeto

```python
from pathlib import Path

from causal_discovery import load_time_series_dataset

dataset = load_time_series_dataset(
    Path("datasets/causaltime/traffic/gen_data.npy"),
    data_format="causaltime",
    graph_path=Path("datasets/causaltime/traffic/graph.npy"),
    trajectory_index=0,
    column_prefix="traffic",
    selected_columns=None,
)

data = dataset.data
ground_truth = dataset.ground_truth
```

Com `selected_columns=None`, todos os 20 nós observados são carregados. Para trabalhar
com um subconjunto explícito:

```python
selected_columns = [
    "traffic_02",
    "traffic_04",
    "traffic_06",
    "traffic_08",
    "traffic_13",
    "traffic_19",
]
```

As opções apresentadas pela interface são geradas dinamicamente a partir das colunas
carregadas. Portanto, trocar `selected_columns` também altera as variáveis disponíveis
para definição dos objetivos da análise.

## Carregamento NumPy direto

Para inspecionar os arquivos sem usar o carregador do projeto:

```python
import numpy as np

generated = np.load("datasets/causaltime/traffic/gen_data.npy")
graph = np.load("datasets/causaltime/traffic/graph.npy")

observed = generated[:, :, : graph.shape[0]]

print(generated.shape)  # (480, 40, 40)
print(observed.shape)   # (480, 40, 20)
print(graph.shape)      # (20, 20)
```

## Limitações de interpretação

- Não existe, neste pacote, um mapeamento dos índices para sensores ou locais reais.
- Os valores não possuem unidade física recuperável apenas pelos arquivos fornecidos.
- O grafo é uma referência usada para geração e avaliação, não uma causalidade real
  comprovada a partir dos dados observacionais originais.
- O ground truth indica adjacência, mas não o lag causal exato.
- Escolher nós com base no próprio grafo é útil para depuração, porém não constitui uma
  avaliação cega do benchmark.

## Referências

- [CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery](https://arxiv.org/html/2310.01753)
- [DCRNN e dados PEMS-BAY](https://github.com/liyaguang/DCRNN)
