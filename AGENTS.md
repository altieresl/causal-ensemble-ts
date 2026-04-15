# AGENTS.md

## Objetivo

Este arquivo define regras para qualquer agente que editar este repositorio.

## Escopo

As regras abaixo valem para todo o projeto, incluindo o notebook e os modulos em `causal_discovery/`.

## Regras de codigo

1. Preservar APIs publicas existentes, a menos que a mudanca seja solicitada explicitamente.
2. Fazer mudancas pequenas, focadas e com impacto local.
3. Evitar refatoracoes amplas quando o pedido for especifico.
4. Manter estilo consistente com o arquivo alterado.
5. Evitar dependencias novas sem necessidade clara.

## Regras de notebook

1. Nao apagar celulas do usuario sem solicitacao explicita.
2. Evitar outputs grandes e sensiveis no notebook versionado.
3. Manter celulas em ordem logica: preparacao, execucao, analise.
4. Preferir codigo reproduzivel (sem caminhos absolutos de maquina).

## Regras de qualidade

1. Validar sintaxe e imports apos alteracoes.
2. Em caso de erro novo causado pela mudanca, corrigir antes de finalizar.
3. Nao alterar arquivos fora do escopo da tarefa.

## Seguranca e privacidade

1. Nao incluir credenciais, tokens ou dados pessoais em codigo, logs ou notebook outputs.
2. Evitar registrar caminhos locais de ambiente quando nao for necessario.

## Comunicacao

1. Explicar brevemente o que foi alterado e por que.
2. Listar qualquer limitacao que tenha impedido validacao completa.
3. Se houver risco de quebra, sinalizar claramente antes de encerrar.

## Git e entrega

1. Nao versionar `.venv/`, caches e artefatos temporarios.
2. Preferir commits pequenos e com mensagem objetiva.
3. Nao reverter mudancas do usuario sem pedido explicito.
