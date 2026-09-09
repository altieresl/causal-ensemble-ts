# causal_algorithms_atlas

Base de conhecimento verificada sobre algoritmos de causal discovery em séries
temporais, independente do pipeline de execução em `causal_discovery/`. Pensada para
alimentar futuramente um sistema de RAG cujo uso principal é **seleção de
algoritmos**: dado o perfil dos dados, recomendar quais métodos compor no ensemble.

## Estrutura

```text
causal_algorithms_atlas/
    schema.py       # vocabulario controlado (enums) e dataclasses tipadas
    loader.py       # parse de algorithms/*.md (frontmatter YAML + prosa)
    validate.py     # validacao cruzada (referencias, alinhamento com o framework)
    export.py       # exporta fichas para chunks JSONL (uso em RAG)
    evidence.py     # parse de evidence/*.yaml (resultados empiricos)
    rag_chat.py      # chat de debug: TF-IDF + Llama local via Ollama
    algorithms/      # uma ficha .md por algoritmo
    evidence/        # resultados empiricos medidos neste projeto, por algoritmo+dataset
    references.yaml # bibliografia compartilhada (DOI/arXiv/venue por chave de citacao)
```

Cada ficha em `algorithms/<id>.md` só existe se tiver referência primária verificável
(DOI, arXiv ou publicação revisada por pares) — ver o campo `verification` e
`verified_by` no frontmatter. `evidence/` é uma camada separada: guarda o que **este
projeto mediu** nos próprios benchmarks, não o que a literatura afirma — as duas coisas
nunca ficam no mesmo arquivo.

## Comandos úteis

Rodar a suíte de testes do atlas:

```bash
python -m pytest tests/test_atlas_schema.py tests/test_atlas_loader.py tests/test_atlas_validate.py tests/test_atlas_content.py tests/test_atlas_export.py tests/test_atlas_evidence.py tests/test_atlas_rag_chat.py -v
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
ollama pull llama3.1:8b
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
python -m causal_algorithms_atlas.rag_chat "Qual metodo tolera confundidor latente?"
```

A saída tem duas partes:

- **chunks recuperados**: os trechos da base mais relevantes para a pergunta, com
  score de similaridade — confira se fazem sentido antes de confiar na resposta;
- **resposta**: o que o Llama gerou com base só nesse contexto (o prompt instrui o
  modelo a não inventar quando o contexto não tiver a resposta).

Faça perguntas em português — a base inteira (prosa) está em PT-BR e a recuperação é
por casamento léxico (TF-IDF), então uma pergunta em inglês não encontra o vocabulário
certo no corpus (ver observações abaixo).

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
