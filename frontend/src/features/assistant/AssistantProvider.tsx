import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { streamAssistant, type AssistantContext, type AssistantSource } from "../../api/assistant";
import { ApiError } from "../../api/client";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: AssistantSource[];
  error?: string;
  pending?: boolean;
}

interface AssistantApi {
  open: boolean;
  setOpen: (open: boolean) => void;
  messages: ChatMessage[];
  busy: boolean;
  /** Envia uma pergunta (abre o painel); o contexto é o que a tela atual informar. */
  ask: (question: string) => void;
  stop: () => void;
  clear: () => void;
  context: AssistantContext | null;
  setContext: (context: AssistantContext | null) => void;
  useContextInPrompt: boolean;
  setUseContextInPrompt: (value: boolean) => void;
}

const AssistantCtx = createContext<AssistantApi | null>(null);

export function useAssistant(): AssistantApi {
  const value = useContext(AssistantCtx);
  if (!value) throw new Error("useAssistant fora do AssistantProvider");
  return value;
}

/** Versão tolerante: componentes reutilizáveis (ex.: testes isolados) funcionam sem o provider. */
export const useOptionalAssistant = () => useContext(AssistantCtx);

const STORAGE_KEY = "causal-discovery-ts.assistant";
const MAX_STORED = 30;

function load(): { messages: ChatMessage[]; open: boolean } {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { messages: [], open: false };
    const parsed = JSON.parse(raw) as { messages?: ChatMessage[]; open?: boolean };
    return { messages: (parsed.messages ?? []).filter((m) => !m.pending), open: Boolean(parsed.open) };
  } catch {
    return { messages: [], open: false };
  }
}

let counter = 0;
const newId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`;

export function AssistantProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(load, []);
  const [open, setOpen] = useState(initial.open);
  const [messages, setMessages] = useState<ChatMessage[]>(initial.messages);
  const [busy, setBusy] = useState(false);
  const [context, setContext] = useState<AssistantContext | null>(null);
  const [useContextInPrompt, setUseContextInPrompt] = useState(true);
  const controller = useRef<AbortController | null>(null);
  const latest = useRef(messages);
  latest.current = messages;

  useEffect(() => {
    try {
      const stored = messages.filter((m) => !m.pending).slice(-MAX_STORED);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ messages: stored, open }));
    } catch {
      /* sem persistência: a conversa vale só nesta aba */
    }
  }, [messages, open]);

  const patchLast = useCallback((patch: (message: ChatMessage) => ChatMessage) => {
    setMessages((current) => {
      const copy = current.slice();
      copy[copy.length - 1] = patch(copy[copy.length - 1]);
      return copy;
    });
  }, []);

  const stop = useCallback(() => controller.current?.abort(), []);

  const ask = useCallback(
    (question: string) => {
      const text = question.trim();
      if (!text || busy) return;
      setOpen(true);
      const history = [...latest.current.filter((m) => !m.error && m.content), { id: newId(), role: "user" as const, content: text }];
      setMessages([...latest.current, history[history.length - 1], { id: newId(), role: "assistant", content: "", pending: true }]);
      setBusy(true);
      const abort = new AbortController();
      controller.current = abort;
      streamAssistant(
        {
          messages: history.slice(-20).map(({ role, content }) => ({ role, content })),
          context: useContextInPrompt ? context : null,
        },
        (event) => {
          if (event.type === "sources") patchLast((m) => ({ ...m, sources: event.sources }));
          else if (event.type === "token") patchLast((m) => ({ ...m, content: m.content + event.content }));
          else if (event.type === "error") patchLast((m) => ({ ...m, error: event.message }));
        },
        abort.signal,
      )
        .catch((error: unknown) => {
          if ((error as Error).name === "AbortError") {
            patchLast((m) => ({ ...m, content: m.content ? `${m.content}\n\n_(resposta interrompida)_` : "", error: m.content ? undefined : "Resposta interrompida." }));
          } else {
            patchLast((m) => ({ ...m, error: error instanceof ApiError || error instanceof Error ? error.message : "Erro inesperado." }));
          }
        })
        .finally(() => {
          patchLast((m) => ({ ...m, pending: false }));
          setBusy(false);
          controller.current = null;
        });
    },
    [busy, context, useContextInPrompt, patchLast],
  );

  const clear = useCallback(() => {
    controller.current?.abort();
    setMessages([]);
  }, []);

  const api = useMemo<AssistantApi>(
    () => ({ open, setOpen, messages, busy, ask, stop, clear, context, setContext, useContextInPrompt, setUseContextInPrompt }),
    [open, messages, busy, ask, stop, clear, context, useContextInPrompt],
  );
  return <AssistantCtx.Provider value={api}>{children}</AssistantCtx.Provider>;
}
