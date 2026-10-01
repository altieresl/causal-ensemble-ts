import { ApiError } from "./client";
import type { components } from "./schema";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export type AssistantRequest = components["schemas"]["AssistantRequest"];
export type AssistantContext = components["schemas"]["AssistantContext"];

export interface AssistantSource {
  algorithm_id: string;
  section: string;
  text: string;
  score: number;
}

export type AssistantEvent =
  | { type: "sources"; sources: AssistantSource[] }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

export interface AssistantStatus {
  available: boolean;
  models: string[];
  default_model: string;
  base_url: string;
  error?: string;
}

/** Separa linhas NDJSON completas; devolve os eventos e o resto incompleto para o próximo pedaço. */
export function parseNdjson(buffer: string): { events: AssistantEvent[]; rest: string } {
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  const events: AssistantEvent[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      events.push(JSON.parse(trimmed) as AssistantEvent);
    } catch {
      /* linha corrompida: ignora sem derrubar a conversa */
    }
  }
  return { events, rest };
}

export async function fetchAssistantStatus(): Promise<AssistantStatus> {
  const response = await fetch(`${BASE}/assistant/status`);
  if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status);
  return (await response.json()) as AssistantStatus;
}

/** Envia a conversa e entrega cada evento do stream assim que chega. */
export async function streamAssistant(
  body: AssistantRequest,
  onEvent: (event: AssistantEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError("Não foi possível contatar a API.", 0);
  }
  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const problem = (await response.json()) as { detail?: string };
      detail = problem.detail ?? detail;
    } catch {
      /* sem corpo JSON */
    }
    throw new ApiError(detail, response.status);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseNdjson(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  const tail = parseNdjson(buffer + "\n");
  tail.events.forEach(onEvent);
}
