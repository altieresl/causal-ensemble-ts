# Causal Algorithms Atlas — Design

**Status:** aprovado pelo usuário para implementação direta (sem gate de revisão adicional, a pedido explícito).

## Objetivo

Construir uma base de conhecimento estruturada sobre algoritmos de causal discovery em
séries temporais, para uso futuro como corpus de um sistema RAG cujo caso de uso
principal é **seleção de algoritmos**: dado o perfil dos dados (linearidade,
estacionariedade, número de variáveis, confundidores latentes suspeitos, tamanho de
amostra), recomendar quais métodos compor no ensemble.

Este documento cobre a base + tooling. EDA sobre a base e o chat de RAG (debug) são
fases subsequentes do mesmo plano, descritas abaixo.

## Restrição inegociável: verificação de procedência

Só entra na base algoritmo com referência primária real e verificável (DOI, arXiv ID
ou publicação em veículo revisado por pares). Nenhum campo é preenchido por inferência
não verificada. Um campo sem suporte fica ausente/`null`, nunca "chutado". Isso vale
mais que atingir uma meta numérica de cobertura — a base cresce até onde a verificação
aguenta, não até um número fixo.

Camadas de verificação por conteúdo:

- **Metadados bibliográficos** (autores, ano, veículo, DOI/arXiv): verificados por
  busca (WebSearch/WebFetch) no momento da escrita de cada ficha.
- **Conteúdo técnico dos 8 métodos já implementados no framework** (`PCMCI`, `LPCMCI`,
  `ClassicalGranger`, `NeuralGrangercMLP`, `VARLiNGAM`, `DYNOTEARS`, `GES`, `FCI`):
  verificado cruzando `.local/METHODS.md`, o código-fonte do wrapper e o paper
  original. Estes recebem `verification: verified`.
- **Conteúdo técnico de métodos adicionais da literatura** (não implementados):
  escrito a partir do paper original (abstract + método, verificado por busca),
  `verification: verified` somente quando a referência primária foi confirmada;
  caso contrário `verification: draft` e o campo `verified_by` fica vazio.

## Nome e estrutura do pacote

Pacote novo, top-level, independente de `causal_discovery/`: **`causal_algorithms_atlas/`**.

Motivo do nome: evita o genérico "kb"; "atlas" comunica "coleção de referência
navegável de algoritmos", que é o papel real do pacote.

```text
causal_algorithms_atlas/
    __init__.py
    schema.py          # dataclasses + enums do vocabulário controlado
    loader.py          # parse de Markdown+frontmatter -> objetos tipados
    validate.py         # regras de validação (vocabulário, obrigatoriedade, gate de verificação)
    export.py          # exporta fichas para JSONL (metadado + chunks de prosa) para ingestão RAG
    cli.py              # `python -m causal_algorithms_atlas.cli validate|export`
    algorithms/         # uma ficha .md por algoritmo (frontmatter YAML + prosa PT-BR)
        pcmci.md
        ...
    evidence/           # camada empírica separada, referenciando algorithm_id
        pcmci__toy_a_linear.yaml
        ...
    references.yaml     # bibliografia compartilhada, chave de citação -> metadados

tests/
    test_atlas_schema.py
    test_atlas_loader.py
    test_atlas_validate.py
    test_atlas_export.py
    test_atlas_content.py   # invariantes sobre o conteúdo real (ex.: todo método do
                             # framework tem ficha verified)
```

Este pacote não depende de `causal_discovery` (evita puxar `torch`/`tigramite` para
validar texto) e `causal_discovery` não depende dele — o único acoplamento é o teste
`test_atlas_content.py`, que importa `causal_discovery.discover_causal_methods()` para
checar que todo método registrado tem ficha `verified`.

## Schema da ficha de algoritmo (`algorithms/<id>.md`)

Frontmatter YAML, campos em inglês (vocabulário controlado), prosa em português.

```yaml
id: pcmci                     # slug, chave estável usada por evidence/ e pelos testes
name: "PCMCI"
aliases: []
family: constraint-based      # enum: constraint-based | score-based | functional-causal-model
                               #   | granger-based | continuous-optimization | information-theoretic | hybrid
temporal_handling: native      # enum: native | windowed-unrolling | segmentation | none
output_type: dag               # enum: dag | pag | signed-graph | unsigned-graph | partial-graph
assumptions:
  - id: causal_sufficiency     # vocabulário controlado (ver schema.py: KNOWN_ASSUMPTIONS)
    required: true
    statement: "Ausência de confundidores latentes não medidos."
  - id: stationarity
    required: true
    statement: "Processo estacionário no intervalo analisado."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null         # null = não determinado/verificado
  sample_type: single-series   # enum: single-series | panel | both
implemented_in_framework: true
framework_method_name: "PCMCI"   # deve casar com o nome no registry quando presente
references: [runge2019]          # chaves de references.yaml
verification: verified           # enum: verified | draft
verified_by: "paper-cross-check+source-code"   # obrigatório quando verification=verified
last_reviewed: "2026-09-07"
```

Seções de prosa fixas no corpo (facilita chunking determinístico para RAG):

```markdown
## Ideia central
## Premissas
## Quando usar
## Quando evitar
## Relação com outros métodos
## Notas de implementação   (só quando implemented_in_framework: true)
```

`schema.py` define os enums e a lista fechada `KNOWN_ASSUMPTIONS` (causal_sufficiency,
stationarity, linearity, acyclicity_instantaneous, non_gaussian_errors, faithfulness,
markov_condition, no_selection_bias) — `validate.py` rejeita qualquer `assumptions[].id`
fora dessa lista, para impedir sinônimos divergentes (`linear` vs `linearity`).

## `references.yaml`

```yaml
runge2019:
  authors: "Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., Sejdinovic, D."
  year: 2019
  title: "Detecting and quantifying causal associations in large nonlinear time series datasets"
  venue: "Science Advances"
  doi: "10.1126/sciadv.aau4996"
  verified: true
```

## Camada empírica (`evidence/`)

Arquivos YAML separados das fichas de literatura, para não misturar "o paper afirma"
com "meu experimento mediu":

```yaml
algorithm_id: pcmci
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
produced_by: ".local/results/toy_synthetic_validation/metrics.csv"  # não versionado; documentativo
metrics:
  precision: 0.75
  recall: 1.0
  f1_score: 0.8571428571428571
notes: >
  Medido no benchmark sintético do projeto (ground truth conhecido), não é resultado
  do paper original.
```

`loader.py` carrega `evidence/*.yaml` e valida que `algorithm_id` existe em
`algorithms/`. O pacote não lê `.local/` em runtime (é gitignored e não confiável);
os valores em `evidence/*.yaml` são copiados manualmente/por script a partir dos
resultados locais no momento em que a ficha é escrita.

## `export.py` — saída para RAG

Gera um `.jsonl` com um registro por chunk:

```json
{"id": "pcmci#quando-usar", "algorithm_id": "pcmci", "section": "Quando usar",
 "text": "...", "metadata": {"family": "constraint-based", "handles_latent_confounders": false, ...}}
```

Um chunk por seção de prosa + um chunk "resumo" sintetizado a partir dos campos
estruturados (para casar perguntas de seleção que citam atributos diretamente, ex.
"método que tolera confundidor latente").

## Cobertura inicial de conteúdo

1. Os 8 métodos do framework (`verified`, prioridade máxima).
2. Métodos adicionais amplamente estabelecidos e de fácil verificação de referência
   primária (aplicados um a um, cada um confirmado por busca antes de escrever a
   ficha): candidatos incluem PC/PC-stable, PCMCI+, Transfer Entropy, Convergent Cross
   Mapping (CCM), TiMINo, NOTEARS, SVAR-FCI/tsFCI, TCDF, Rhino. Nem todos
   necessariamente entram — cada um só entra se a verificação passar; o número final é
   resultado da curadoria, não uma meta.

## Fase 2 — EDA sobre a base

Objetivo: analisar a própria base de conhecimento (não os datasets de séries
temporais) — cobertura por família, por premissa, combinações de atributos
cobertas/faltantes. Usa `pandas` + `plotly` (ambos já em `requirements.txt`; não é
necessária nenhuma biblioteca nova — "eda-viz"/"python-eda-viz" não existem como
pacotes mantidos no PyPI, e adicionar uma dependência nova sem necessidade clara viola
a regra 5 de `AGENTS.md`).

Entregável: `causal_algorithms_atlas/eda.py` com funções que carregam todas as fichas
via `loader.py` e produzem um `pandas.DataFrame` tabular (uma linha por algoritmo) mais
gráficos plotly (contagem por família, matriz booleana de premissas por algoritmo,
cobertura de `handles_latent_confounders` x `family`). Script executável
(`python -m causal_algorithms_atlas.eda`) que salva os gráficos como HTML em
`causal_algorithms_atlas/eda_output/` (gitignored, é artefato gerado).

## Fase 3 — Chat RAG de debug com Llama local

Objetivo: consulta exploratória simples para o usuário observar o pipeline
funcionando — não é o produto final de RAG do projeto de dissertação.

- **Modelo:** Ollama já está instalado localmente (`ollama.exe`, v0.17.4). Hardware
  local: RTX 3070 (8 GB VRAM) + 32 GB RAM — não comporta Llama 4 Scout/Maverick nem
  Llama 3.3 70B com qualidade aceitável. Modelo escolhido: **`llama3.1:8b-instruct`**
  (via `ollama pull`), o Llama estável mais recente que roda confortavelmente neste
  hardware. Llama 4 e 3.3-70B ficam registrados como opção futura se o usuário migrar
  para hardware maior.
- **Retrieval:** TF-IDF + similaridade de cosseno via `scikit-learn`
  (`TfidfVectorizer`, `cosine_similarity`) sobre os chunks exportados por `export.py`.
  Já é dependência do projeto — evita adicionar `sentence-transformers`/`faiss` só para
  uma consulta de debug.
- **Geração:** chamada HTTP à API local do Ollama (`http://localhost:11434/api/generate`)
  usando `urllib.request` da stdlib — evita adicionar `requests` como dependência nova.
- **Entregável:** `causal_algorithms_atlas/rag_chat.py`, executável via
  `python -m causal_algorithms_atlas.rag_chat "pergunta"`. Imprime: os top-k chunks
  recuperados (com `algorithm_id` e score) e a resposta gerada pelo modelo, para que o
  usuário valide manualmente que a recuperação faz sentido antes de qualquer uso sério.

## Nova dependência

`pyyaml` — necessária para o parsing do frontmatter. Nenhuma outra dependência nova é
introduzida (TF-IDF vem de `scikit-learn`, gráficos vêm de `plotly`, chamada HTTP vem
da stdlib).

## Testes

- `test_atlas_schema.py`: enums rejeitam valores fora do vocabulário controlado;
  `verification: verified` exige `verified_by` não vazio.
- `test_atlas_loader.py`: parse de frontmatter + prosa em seções nomeadas; erro claro
  em YAML malformado ou seção obrigatória ausente.
- `test_atlas_validate.py`: `assumptions[].id` fora de `KNOWN_ASSUMPTIONS` falha;
  `references` apontando para chave inexistente em `references.yaml` falha;
  `framework_method_name`, quando presente, deve casar com uma chave retornada por
  `causal_discovery.discover_causal_methods()` — nome divergente é erro de validação.
- `test_atlas_content.py`: todo método retornado por
  `causal_discovery.discover_causal_methods()` tem ficha correspondente com
  `verification: verified`; toda ficha `evidence/*.yaml` referencia um `algorithm_id`
  existente; todo `references[]` de toda ficha existe em `references.yaml`.
- `test_atlas_export.py`: cada seção de prosa vira exatamente um chunk; chunk de
  resumo contém os campos estruturados centrais.

## Fora de escopo (explicitamente adiado)

- Vector store persistente / embeddings densos — TF-IDF é suficiente para a consulta
  de debug.
- Interface web para o chat — CLI é suficiente por ora.
- Ingestão automática de PDFs de papers.
- Meta fixa de "30+" algoritmos — superada pela restrição de verificação.
