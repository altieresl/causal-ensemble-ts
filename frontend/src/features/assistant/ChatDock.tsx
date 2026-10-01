import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { matchPath, useLocation } from "react-router-dom";

import { fetchAssistantStatus } from "../../api/assistant";
import { useExperience } from "../../lib/experience";
import { Markdown } from "../../lib/markdown";
import { useAssistant, type ChatMessage } from "./AssistantProvider";

/** Contexto da tela atual (o que o assistente recebe): dataset ou execução aberta. */
function useScreenContext() {
  const { pathname } = useLocation();
  const dataset = matchPath("/datasets/:id/*", pathname) ?? matchPath("/datasets/:id", pathname);
  if (dataset?.params.id) return { kind: "dataset" as const, id: dataset.params.id };
  const run = matchPath("/runs/:id", pathname);
  if (run?.params.id) return { kind: "run" as const, id: run.params.id };
  return null;
}

function suggestionsFor(kind: "dataset" | "run" | null, beginner: boolean): string[] {
  if (kind === "dataset") {
    return beginner
      ? ["O que esse perfil diz sobre os meus dados?", "Por que alguns métodos violam premissas?", "Por onde começo a análise?"]
      : ["Quais métodos são mais adequados a este perfil e por quê?", "Os métodos com premissa violada devem ficar no ensemble?", "Qual lag máximo faz sentido aqui?"];
  }
  if (kind === "run") {
    return beginner
      ? ["Explique este resultado em linguagem simples.", "O que significa a probabilidade de uma aresta?", "Posso confiar nessas relações?"]
      : ["Quais arestas merecem mais cautela e por quê?", "Por que essa combinação de métodos foi escolhida?", "Que análise complementar você sugere?"];
  }
  return [
    "Qual a diferença entre PCMCI e Granger?",
    "Quais métodos toleram confundidores latentes?",
    "Quais métodos assumem estacionariedade?",
  ];
}

function Message({ message }: { message: ChatMessage }) {
  const [showSources, setShowSources] = useState(false);
  if (message.role === "user") {
    return (
      <div className="msg user">
        <p>{message.content}</p>
      </div>
    );
  }
  return (
    <div className="msg assistant" aria-busy={message.pending}>
      {message.content ? <Markdown text={message.content} /> : message.pending && !message.error && <span className="typing" aria-label="Escrevendo">●●●</span>}
      {message.error && (
        <p className="msg-error" role="alert">
          {message.error}
        </p>
      )}
      {message.sources && message.sources.length > 0 && (
        <div className="msg-sources">
          <button type="button" className="link small" onClick={() => setShowSources(!showSources)} aria-expanded={showSources}>
            {showSources ? "Ocultar" : "Ver"} fontes do atlas ({message.sources.length})
          </button>
          {showSources && (
            <ul className="plain">
              {message.sources.map((source, index) => (
                <li key={index}>
                  <strong>
                    {source.algorithm_id} · {source.section}
                  </strong>
                  <p className="muted small">{source.text.length > 280 ? `${source.text.slice(0, 280)}…` : source.text}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/** Painel do assistente fixo na lateral; acompanha a navegação e sabe o que está na tela. */
export function ChatDock() {
  const assistant = useAssistant();
  const { isBeginner } = useExperience();
  const screen = useScreenContext();
  const [draft, setDraft] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const status = useQuery({
    queryKey: ["assistant", "status"],
    queryFn: fetchAssistantStatus,
    enabled: assistant.open,
    refetchInterval: assistant.open ? 30_000 : false,
    retry: 0,
  });

  const { setContext, open, setOpen } = assistant;
  const { search } = useLocation();
  // deep-link: qualquer URL com ?assistente=1 abre o painel
  useEffect(() => {
    if (new URLSearchParams(search).get("assistente") === "1") setOpen(true);
  }, [search, setOpen]);
  const screenKey = screen ? `${screen.kind}:${screen.id}` : "";
  useEffect(() => {
    setContext(screen);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screenKey, setContext]);

  // empurra o conteúdo para o lado em telas largas, em vez de cobri-lo
  useEffect(() => {
    document.body.classList.toggle("assistant-open", open);
    return () => document.body.classList.remove("assistant-open");
  }, [open]);

  useEffect(() => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight; // acompanha a resposta enquanto ela chega
  }, [assistant.messages]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const send = () => {
    if (!draft.trim() || assistant.busy) return;
    assistant.ask(draft);
    setDraft("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  if (!open) {
    return (
      <button type="button" className="assistant-launcher" onClick={() => assistant.setOpen(true)} aria-label="Abrir o assistente">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path d="M4 5h16v11H8l-4 4V5z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M8 9h8M8 12h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
        Assistente
      </button>
    );
  }

  const offline = status.data && !status.data.available;
  const contextLabel = screen ? (screen.kind === "dataset" ? `dataset ${screen.id}` : `execução ${screen.id}`) : null;

  return (
    <aside
      className="assistant-panel"
      aria-label="Assistente"
      onKeyDown={(event) => {
        if (event.key === "Escape") assistant.setOpen(false);
      }}
    >
      <header className="assistant-header">
        <div>
          <strong>Assistente</strong>
          <span className="muted small">
            {" "}
            · {status.data?.available ? status.data.default_model : status.isPending ? "verificando…" : "offline"}
          </span>
        </div>
        <div className="row" style={{ gap: ".25rem" }}>
          <button type="button" className="ghost small" onClick={assistant.clear} disabled={assistant.messages.length === 0} title="Apagar a conversa">
            Limpar
          </button>
          <button type="button" className="ghost" onClick={() => assistant.setOpen(false)} aria-label="Fechar o assistente" title="Fechar (Esc)">
            ×
          </button>
        </div>
      </header>

      {offline && (
        <p className="assistant-banner" role="status">
          O Ollama não respondeu em {status.data?.base_url}. Rode <code>ollama serve</code> e{" "}
          <code>ollama pull {status.data?.default_model}</code> na máquina da API. Até lá, as respostas trazem só as fontes do atlas.
        </p>
      )}

      {contextLabel && (
        <label className="assistant-context small">
          <input
            type="checkbox"
            checked={assistant.useContextInPrompt}
            onChange={(e) => assistant.setUseContextInPrompt(e.target.checked)}
          />
          Usar o que está na tela: <strong>{contextLabel}</strong>
        </label>
      )}

      <div className="assistant-messages" ref={listRef} aria-live="polite">
        {assistant.messages.length === 0 ? (
          <div className="assistant-empty">
            <p>
              Pergunte sobre os algoritmos, o perfil dos dados ou o resultado aberto. As respostas usam as fichas do atlas e
              o que está na tela — nunca o grafo verdadeiro.
            </p>
            <div className="stack-sm">
              {suggestionsFor(screen?.kind ?? null, isBeginner).map((suggestion) => (
                <button key={suggestion} type="button" className="suggestion" onClick={() => assistant.ask(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          assistant.messages.map((message) => <Message key={message.id} message={message} />)
        )}
      </div>

      <form
        className="assistant-input"
        onSubmit={(event) => {
          event.preventDefault();
          send();
        }}
      >
        <textarea
          ref={inputRef}
          rows={2}
          value={draft}
          maxLength={4000}
          placeholder="Escreva sua pergunta… (Enter envia, Shift+Enter quebra linha)"
          aria-label="Pergunta para o assistente"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {assistant.busy ? (
          <button type="button" onClick={assistant.stop}>
            Parar
          </button>
        ) : (
          <button type="submit" className="primary" disabled={!draft.trim()}>
            Enviar
          </button>
        )}
      </form>
    </aside>
  );
}
