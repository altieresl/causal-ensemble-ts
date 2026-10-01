import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { parseNdjson } from "../../api/assistant";
import { ExperienceProvider } from "../../lib/experience";
import { Markdown } from "../../lib/markdown";
import { AssistantProvider } from "./AssistantProvider";
import { ChatDock } from "./ChatDock";

function ndjsonResponse(lines: object[], chunkSize = 7) {
  const text = lines.map((line) => JSON.stringify(line)).join("\n") + "\n";
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      // pedaços pequenos para exercitar linhas quebradas no meio
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
      controller.close();
    },
  });
  return { ok: true, status: 200, statusText: "OK", body, json: async () => ({}) };
}

function setup(path = "/datasets/toy_a", chatLines?: object[]) {
  const calls: { url: string; body?: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith("/assistant/status")) {
        return { ok: true, status: 200, json: async () => ({ available: true, models: ["m"], default_model: "m", base_url: "x" }) };
      }
      return ndjsonResponse(
        chatLines ?? [
          { type: "sources", sources: [{ algorithm_id: "pcmci", section: "Assumptions", text: "Assume estacionariedade.", score: 0.4 }] },
          { type: "token", content: "O **PCMCI** " },
          { type: "token", content: "assume estacionariedade." },
          { type: "done" },
        ],
      );
    }),
  );
  window.localStorage.clear();
  window.localStorage.setItem("causal-discovery-ts.experience", "advanced");
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ExperienceProvider>
        <MemoryRouter initialEntries={[path]}>
          <AssistantProvider>
            <ChatDock />
          </AssistantProvider>
        </MemoryRouter>
      </ExperienceProvider>
    </QueryClientProvider>,
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("parseNdjson", () => {
  it("devolve eventos completos e guarda o resto incompleto", () => {
    const { events, rest } = parseNdjson('{"type":"done"}\n{"type":"tok');
    expect(events).toEqual([{ type: "done" }]);
    expect(rest).toBe('{"type":"tok');
  });
  it("ignora linhas corrompidas", () => {
    expect(parseNdjson('lixo\n{"type":"done"}\n').events).toEqual([{ type: "done" }]);
  });
});

describe("Markdown", () => {
  it("renderiza listas e negrito sem interpretar HTML", () => {
    const { container } = render(<Markdown text={"**Atenção**\n\n- um\n- <img src=x onerror=alert(1)>"} />);
    expect(container.querySelector("strong")?.textContent).toBe("Atenção");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });
});

describe("ChatDock", () => {
  it("abre, envia a pergunta com o contexto da tela e mostra a resposta em streaming", async () => {
    const calls = setup();
    fireEvent.click(screen.getByRole("button", { name: "Abrir o assistente" }));
    expect(screen.getByText("dataset toy_a")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Pergunta para o assistente"), { target: { value: "Quais métodos usar?" } });
    fireEvent.keyDown(screen.getByLabelText("Pergunta para o assistente"), { key: "Enter" });

    await waitFor(() => expect(screen.getByText("PCMCI")).toBeInTheDocument());
    expect(screen.getByText(/assume estacionariedade\./)).toBeInTheDocument();
    const chat = calls.find((c) => c.url.endsWith("/assistant/chat"))!;
    expect(chat.body).toEqual({
      messages: [{ role: "user", content: "Quais métodos usar?" }],
      context: { kind: "dataset", id: "toy_a" },
    });

    fireEvent.click(screen.getByRole("button", { name: /Ver fontes do atlas \(1\)/ }));
    expect(screen.getByText("pcmci · Assumptions")).toBeInTheDocument();
  });

  it("desligar o contexto não envia o dataset; sugestões perguntam com um clique", async () => {
    const calls = setup("/runs/run_abc");
    fireEvent.click(screen.getByRole("button", { name: "Abrir o assistente" }));
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "Por que essa combinação de métodos foi escolhida?" }));
    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/assistant/chat"))).toBe(true));
    expect(calls.find((c) => c.url.endsWith("/assistant/chat"))!.body).toMatchObject({ context: null });
  });

  it("mostra o erro do servidor (Ollama fora do ar) sem perder a pergunta", async () => {
    setup("/", [
      { type: "sources", sources: [] },
      { type: "error", message: "Nao foi possivel falar com o Ollama" },
    ]);
    fireEvent.click(screen.getByRole("button", { name: "Abrir o assistente" }));
    fireEvent.click(screen.getByRole("button", { name: "Qual a diferença entre PCMCI e Granger?" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Ollama");
    expect(screen.getByText("Qual a diferença entre PCMCI e Granger?", { selector: ".msg.user p" })).toBeInTheDocument();
  });

  it("guarda a conversa no navegador e fecha com Esc", async () => {
    setup();
    fireEvent.click(screen.getByRole("button", { name: "Abrir o assistente" }));
    fireEvent.click(screen.getByRole("button", { name: "Quais métodos são mais adequados a este perfil e por quê?" }));
    await waitFor(() => expect(screen.getByText(/assume estacionariedade\./)).toBeInTheDocument());
    await waitFor(() => {
      const stored = JSON.parse(window.localStorage.getItem("causal-discovery-ts.assistant") ?? "{}");
      expect(stored.messages).toHaveLength(2);
    });
    fireEvent.keyDown(screen.getByLabelText("Pergunta para o assistente"), { key: "Escape" });
    expect(screen.getByRole("button", { name: "Abrir o assistente" })).toBeInTheDocument();
  });
});
