# causal_algorithms_atlas

Base de conhecimento verificada sobre algoritmos de causal discovery em séries
temporais, independente do pipeline de execução em `causal_discovery/`. Pensada para
alimentar futuramente um sistema de RAG cujo uso principal é **seleção de
algoritmos**: dado o perfil dos dados, recomendar quais métodos compor no ensemble.

## Estrutura

```text
causal_algorithms_atlas/
    schema.py             # vocabulario controlado (enums) e dataclasses tipadas
    loader.py             # parse de algorithms/*.md (frontmatter YAML + prosa)
    validate.py           # validacao cruzada (referencias, alinhamento com o framework)
    export.py             # exporta fichas para chunks JSONL (uso em RAG)
    evidence.py           # parse de evidence/*.yaml (resultados empiricos)
    dataset_profile.py    # EDA: extrai um perfil estatistico objetivo de um dataset
    ensemble_advisor.py   # recomendacao deterministica de metodos a partir do perfil
    chat_recommender.py   # recomendacao via LLM local, uma decisao por algoritmo
    experiment_runner.py  # pipeline completo: perfilar -> recomendar -> selecionar -> avaliar
    rag_chat.py           # chat de debug: TF-IDF + LLM local via Ollama
    algorithms/           # uma ficha .md por algoritmo
    evidence/             # resultados empiricos medidos neste projeto, por algoritmo+dataset
    references.yaml       # bibliografia compartilhada (DOI/arXiv/venue por chave de citacao)
```

Cada ficha em `algorithms/<id>.md` só existe se tiver referência primária verificável
(DOI, arXiv ou publicação revisada por pares) — ver o campo `verification` e
`verified_by` no frontmatter. `evidence/` é uma camada separada: guarda o que **este
projeto mediu** nos próprios benchmarks, não o que a literatura afirma — as duas coisas
nunca ficam no mesmo arquivo.

## Do perfil estatístico à seleção de métodos

Esta é a parte do módulo que decide quais dos métodos implementados em
`causal_discovery/` compõem o ensemble para um dataset específico. O princípio
estrutural que atravessa as quatro camadas abaixo: **a seleção nunca consulta
`ground_truth`** — só características objetivas do dataset e premissas declaradas nas
fichas do catálogo. Isso é o que torna válida qualquer comparação posterior entre o
ensemble e um método individual, ou entre a recomendação e o filtro estatístico.

### 1. `dataset_profile.profile_dataset(data) -> DatasetProfile`

Extrai, por variável, três características testáveis a partir dos dados observados —
nunca um valor inferido ou assumido:

| Característica | Método | Referência |
|---|---|---|
| Estacionariedade | Teste ADF (alfa=0,05) | Dickey & Fuller, 1979 |
| Linearidade | Tamanho de efeito: ganho de erro de previsão fora da amostra ao permitir termos quadráticos/cúbicos sobre o lag 1 (janelas expansivas, regressão ridge), contra um limiar calibrado — não um teste de significância pura | metodologia própria do projeto, calibrada com datasets de referência conhecidos |
| Não-gaussianidade dos erros | Tamanho de efeito: skewness/curtose em excesso dos resíduos de um VAR(1), contra um limiar calibrado | idem |

Os dois testes de tamanho de efeito são deliberadamente **não** testes de significância
clássicos (RESET, Shapiro-Wilk): ambos rejeitam a hipótese nula para qualquer desvio,
por menor que seja, assim que a amostra é grande o suficiente, o que os torna
inutilizáveis em datasets com muitas observações. Os limiares usados aqui foram
calibrados observando o "chão de ruído" em datasets com a propriedade oposta conhecida.

Cada campo por variável fica `None`, em vez de assumir um default, quando a série é
curta demais para o teste correspondente. Os agregados (`stationary_fraction`,
`linear_fraction`, `non_gaussian_fraction`) viram booleanos de maioria simples
(`mostly_*`, corte em 50%).

Confundidores latentes (suficiência causal) **não** aparecem em `DatasetProfile`: não
são verificáveis a partir só dos dados observados — é um limite de identificabilidade
(Spirtes, Glymour & Scheines, *Causation, Prediction, and Search*, 2000, Cap. 6), não
uma limitação de metodologia a ser resolvida com um teste melhor.

### 2. `ensemble_advisor.recommend_framework_methods(profile, ...) -> list[MethodRecommendation]`

Filtro determinístico: cruza o perfil acima com as premissas declaradas em cada ficha
`verified` do catálogo (`loader.load_algorithm_cards`). Um método é excluído quando
exige uma premissa (estacionariedade, linearidade ou não-gaussianidade dos erros) que a
maioria do dataset não satisfaz. Aceita um parâmetro opcional
`declared_causal_sufficiency: bool | None`: como suficiência causal não é medível,
esse valor só pode vir de conhecimento de domínio declarado sobre um dataset **real**
(nunca do gerador de um dataset sintético, o que reintroduziria a resposta certa por
outra via) — `None` (padrão) não filtra nada, só anota que a premissa é desconhecida.
`handles_latent_confounders` (campo do catálogo) aparece como nota informativa para os
métodos que lidam nativamente com confundidores não verificados (FCI, LPCMCI — saída em
PAG, Spirtes, Meek & Richardson, 1995), nunca como filtro.

Duas funções traduzem essas recomendações em candidatos executáveis:

- `select_candidate_methods`: filtro **rígido** — só devolve métodos sem nenhuma
  premissa violada. Determinístico e reprodutível; é a versão usada como referência
  para avaliar a qualidade do `chat_recommender`.
- `select_candidate_methods_with_assumption_flags`: pool **ampliado** — todo método
  `verified`+implementado entra como candidato, mesmo violando uma premissa, com a
  violação anexada como flag visível (nunca escondida). Quem decide se um candidato
  flagueado sobrevive é a métrica cega de estabilidade sob bootstrap de
  `causal_discovery.select_robust_ensemble_combination`, nunca a flag em si nem
  `ground_truth`.

### 3. `chat_recommender.recommend_methods_via_chat(profile, ...) -> ChatMethodSelection`

Mesma pergunta que a camada 2, respondida por um LLM local (via Ollama) em vez de uma
regra fixa — para medir se o modelo reproduz a mesma seleção. Uma chamada por
algoritmo (não uma lista única), cada uma restrita à ficha de um único método e ao
perfil do dataset; a resposta é forçada contra um JSON Schema com campos intermediários
(as porcentagens relevantes e as premissas do método, antes da decisão final), e o
Python confere essas porcentagens contra o perfil real e a consistência lógica da
decisão antes de aceitar a resposta — com uma tentativa de repetição automática se
alguma checagem falhar. Nunca recebe `ground_truth` nem vê a saída de
`recommend_framework_methods`.

### 4. `experiment_runner.run_experiment(data, ground_truth=..., ...)`

Amarra as três camadas anteriores num ciclo completo: perfilar → recomendar → montar
o pool de candidatos → selecionar a combinação via
`causal_discovery.select_robust_ensemble_combination` (cega, bootstrap) → só então
consultar `ground_truth` para relatar métricas pós-hoc (precision/recall/F1), nunca
para escolher. `use_assumption_soft_filter` (default `True`) alterna entre as duas
funções de seleção da camada 2.

## Comandos úteis

Rodar a suíte de testes do atlas:

```bash
python -m pytest tests/ -k atlas -v
```

Conferir que todo método registrado em `causal_discovery` tem ficha `verified` no atlas:

```bash
python -m pytest tests/test_atlas_content.py -v
```

## Como rodar o chat RAG de debug

O chat (`rag_chat.py`) recupera os chunks mais relevantes da base via TF-IDF
(`scikit-learn`) e gera uma resposta com um Llama rodando localmente via
[Ollama](https://ollama.com). É uma consulta exploratória — serve para você inspecionar
manualmente se a recuperação faz sentido, não é o RAG final do projeto.

**1. Instale as dependências Python** (se ainda não instalou):

```bash
pip install -r requirements.txt
```

**2. Instale o Ollama**, se ainda não tiver: https://ollama.com/download

**3. Baixe o modelo** (uma vez só; ~5 GB):

```bash
ollama pull qwen2.5:7b
```

**4. Garanta que o serviço do Ollama está rodando.** No Windows, normalmente o app já
mantém o serviço ativo em segundo plano depois de instalado/aberto uma vez. Se o
`rag_chat.py` reclamar de conexão recusada, suba manualmente em um terminal:

```bash
ollama serve
```

(deixe esse terminal aberto; em outro terminal, siga para o passo 5)

**5. Rode a consulta**, a partir da raiz do projeto:

```bash
python -m causal_algorithms_atlas.rag_chat "sua pergunta aqui"
```

Exemplo:

```bash
python -m causal_algorithms_atlas.rag_chat "Which method tolerates latent confounders?"
```

A saída tem duas partes:

- **chunks recuperados**: os trechos da base mais relevantes para a pergunta, com
  score de similaridade — confira se fazem sentido antes de confiar na resposta;
- **resposta**: o que o Llama gerou com base só nesse contexto (o prompt instrui o
  modelo a não inventar quando o contexto não tiver a resposta).

Faça perguntas em inglês — a base inteira (prosa das fichas em `algorithms/*.md`) está
em inglês e a recuperação é por casamento léxico (TF-IDF), então uma pergunta em outro
idioma não encontra o vocabulário certo no corpus (ver observações abaixo).

### Limitações conhecidas desta versão de debug

- **TF-IDF é léxico, não semântico.** Recupera por sobreposição de palavras, não por
  significado. Perguntas com vocabulário muito diferente do texto das fichas (ou em
  outro idioma) tendem a recuperar chunks irrelevantes.
- **`evidence/` (resultados empíricos) ainda não entra no chat.** Só as fichas de
  literatura em `algorithms/` viram chunks hoje; os resultados medidos nos benchmarks
  sintéticos não fazem parte do contexto ainda.
- **Sem histórico de conversa.** Cada chamada é uma pergunta isolada, sem memória da
  pergunta anterior.

## Adicionando um novo algoritmo

1. Confirme uma referência primária real (DOI, arXiv ou venue revisado por pares) —
   nunca escreva uma ficha sem isso.
2. Adicione a referência em `references.yaml`.
3. Crie `algorithms/<id>.md` seguindo o frontmatter e as 5 seções obrigatórias de
   qualquer ficha existente (ex.: `algorithms/pcmci.md`). Use `verification: draft` e
   `verified_by: null` se o conteúdo técnico ainda não foi cruzado com o paper original.
4. Rode `python -m pytest tests/test_atlas_content.py tests/test_atlas_validate.py -v`
   para confirmar que as referências resolvem e o schema aceita a ficha.
