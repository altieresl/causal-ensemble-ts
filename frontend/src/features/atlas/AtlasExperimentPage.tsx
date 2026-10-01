import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { useCreateAtlasChat, useCreateAtlasExperiment, useDataset } from "../../api/hooks";
import { Card, ErrorBox, PageHeader, PageSkeleton } from "../../components/ui";

type Sufficiency = "unknown" | "yes" | "no";
const SUFFICIENCY: Record<Sufficiency, boolean | null> = { unknown: null, yes: true, no: false };

function SufficiencySelect({ value, onChange }: { value: Sufficiency; onChange: (value: Sufficiency) => void }) {
  return (
    <label>
      Suficiência causal declarada por especialista
      <select value={value} onChange={(e) => onChange(e.target.value as Sufficiency)}>
        <option value="unknown">Não sei (padrão)</option>
        <option value="yes">Sim</option>
        <option value="no">Não, há confundidores latentes</option>
      </select>
    </label>
  );
}

/** Experimento do atlas: perfil → métodos compatíveis → seleção cega → avaliação pós-hoc. */
export function AtlasExperimentPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const dataset = useDataset(id);
  const create = useCreateAtlasExperiment();
  const [nBootstrap, setNBootstrap] = useState(3);
  const [maxMethods, setMaxMethods] = useState(3);
  const [maxLag, setMaxLag] = useState(1);
  const [maxRows, setMaxRows] = useState("");
  const [softFilter, setSoftFilter] = useState(true);
  const [sufficiency, setSufficiency] = useState<Sufficiency>("unknown");

  if (dataset.isPending) return <PageSkeleton />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    create.mutate(
      {
        dataset_id: id,
        n_bootstrap: nBootstrap,
        max_methods: maxMethods,
        max_lag: maxLag,
        max_rows: maxRows ? Number(maxRows) : null,
        use_assumption_soft_filter: softFilter,
        declared_causal_sufficiency: SUFFICIENCY[sufficiency],
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <PageHeader
        title="Experimento do atlas"
        crumbs={[{ label: "Datasets", to: "/" }, { label: dataset.data.entry.name, to: `/datasets/${id}` }, { label: "Experimento do atlas" }]}
        lead={
          <>
        Perfila o dataset, escolhe candidatos pelas premissas declaradas nas fichas verificadas e seleciona a melhor
        combinação com métricas cegas ao gabarito. O grafo verdadeiro (se houver) só é consultado depois.
          </>
        }
      />
      <Card title="Parâmetros">
        <div className="form-grid">
          <label>
            Bootstraps
            <input type="number" min={1} max={100} value={nBootstrap} onChange={(e) => setNBootstrap(Number(e.target.value))} />
          </label>
          <label>
            Máximo de métodos por combinação
            <input type="number" min={2} max={20} value={maxMethods} onChange={(e) => setMaxMethods(Number(e.target.value))} />
          </label>
          <label>
            Lag máximo
            <input type="number" min={1} max={20} value={maxLag} onChange={(e) => setMaxLag(Number(e.target.value))} />
          </label>
          <label>
            Limitar linhas (vazio = todas)
            <input type="number" min={50} value={maxRows} onChange={(e) => setMaxRows(e.target.value)} />
          </label>
          <SufficiencySelect value={sufficiency} onChange={setSufficiency} />
        </div>
        <label className="check">
          <input type="checkbox" checked={softFilter} onChange={(e) => setSoftFilter(e.target.checked)} />
          Filtro suave: manter métodos que violam uma premissa (a estabilidade sob bootstrap decide)
        </label>
      </Card>
      {create.isError && <ErrorBox error={create.error} />}
      <div>
        <button type="submit" className="primary" disabled={create.isPending}>
          {create.isPending ? "Enviando…" : "Executar experimento"}
        </button>
      </div>
    </form>
  );
}

/** Seleção de métodos via chat local (Ollama), para comparar com o filtro estatístico. */
export function AtlasChatPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const dataset = useDataset(id);
  const create = useCreateAtlasChat();
  const [model, setModel] = useState("");
  const [parallelCalls, setParallelCalls] = useState(4);
  const [sufficiency, setSufficiency] = useState<Sufficiency>("unknown");

  if (dataset.isPending) return <PageSkeleton />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    create.mutate(
      {
        dataset_id: id,
        model: model.trim() || null,
        max_retries: 1,
        parallel_calls: parallelCalls,
        declared_causal_sufficiency: SUFFICIENCY[sufficiency],
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <PageHeader
        title="Seleção via chat"
        crumbs={[{ label: "Datasets", to: "/" }, { label: dataset.data.entry.name, to: `/datasets/${id}` }, { label: "Seleção via chat" }]}
        lead={
          <>
        Pede a um LLM local (Ollama) que decida, um algoritmo por vez, quais métodos entram, a partir do mesmo perfil e
        das mesmas premissas do filtro estatístico — sem acesso ao gabarito. Requer o serviço <code>ollama serve</code>{" "}
        e o modelo baixado no servidor da API.
          </>
        }
      />
      <Card title="Parâmetros">
        <div className="form-grid">
          <label>
            Modelo (vazio = padrão do servidor)
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="qwen2.5:7b" />
          </label>
          <label>
            Chamadas simultâneas ao Ollama ({parallelCalls})
            <input type="range" min={1} max={8} value={parallelCalls} onChange={(e) => setParallelCalls(Number(e.target.value))} />
            <span className="muted small">Cada algoritmo é uma chamada independente; o ganho depende de OLLAMA_NUM_PARALLEL.</span>
          </label>
          <SufficiencySelect value={sufficiency} onChange={setSufficiency} />
        </div>
      </Card>
      {create.isError && <ErrorBox error={create.error} />}
      <div>
        <button type="submit" className="primary" disabled={create.isPending}>
          {create.isPending ? "Enviando…" : "Consultar o chat"}
        </button>
      </div>
    </form>
  );
}
