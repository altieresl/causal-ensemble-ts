import { Link, useNavigate, useParams } from "react-router-dom";

import { isTerminal, useDeleteRun, useRun, useRunResult } from "../../api/hooks";
import type {
  AtlasExperimentResult,
  BenchmarkResult,
  ChatResult,
  Run,
  RunResult,
  ReplicatedResult,
} from "../../api/types";
import { Card, ErrorBox, Spinner, StatusBadge } from "../../components/ui";
import { formatDateTime, formatDuration } from "../../lib/format";
import { AtlasExperimentView, ChatResultView } from "../results/AtlasResultViews";
import { BenchmarkResultView } from "../results/BenchmarkResultView";
import { ReplicatedResultView } from "../results/ReplicatedResultView";
import { ResultView } from "../results/ResultView";
import { KIND_LABELS } from "./kinds";

function Progress({ progress }: { progress: NonNullable<Run["progress"]> }) {
  const { done, total, message } = progress as { done: number; total: number; message: string };
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  return (
    <div>
      <progress max={total || 1} value={done} aria-label="Progresso da execução" />{" "}
      <span className="muted">
        {message} — {done}/{total} ({percent}%)
      </span>
    </div>
  );
}

function RunResultView({ run }: { run: Run }) {
  const result = useRunResult<unknown>(run.id, true);
  if (result.isPending) return <Spinner label="Carregando resultado…" />;
  if (result.isError) return <ErrorBox error={result.error} />;
  switch (run.kind) {
    case "pipeline":
      return <ResultView result={result.data as RunResult} />;
    case "benchmark":
      return <BenchmarkResultView result={result.data as BenchmarkResult} />;
    case "replicated_validation":
      return <ReplicatedResultView result={result.data as ReplicatedResult} />;
    case "atlas_experiment":
      return <AtlasExperimentView result={result.data as AtlasExperimentResult} />;
    case "atlas_chat":
      return <ChatResultView result={result.data as ChatResult} />;
  }
}

export function RunPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const run = useRun(id);
  const remove = useDeleteRun();

  if (run.isPending) return <Spinner />;
  if (run.isError) return <ErrorBox error={run.error} />;
  const data = run.data;
  const active = !isTerminal(data.status);
  const datasetId = typeof data.params.dataset_id === "string" ? data.params.dataset_id : null;

  return (
    <div className="stack">
      <h1>
        {KIND_LABELS[data.kind]} — {data.id} <StatusBadge status={data.status} />
      </h1>
      <p className="muted">
        {datasetId && (
          <>
            Dataset <Link to={`/datasets/${datasetId}`}>{datasetId}</Link> ·{" "}
          </>
        )}
        criada {formatDateTime(data.created_at)} · duração {formatDuration(data.started_at, data.finished_at)}
      </p>

      {active && (
        <Card>
          <Spinner label={data.status === "queued" ? "Na fila…" : "Executando (pode levar minutos)…"} />
          {data.progress && <Progress progress={data.progress} />}
          <div>
            <button
              className="danger"
              onClick={() => remove.mutate(id)}
              disabled={remove.isPending}
              title="Interrompe no próximo ponto de progresso; o resultado é descartado."
            >
              Cancelar
            </button>
          </div>
        </Card>
      )}
      {data.status === "failed" && (
        <Card title="Falha">
          <p className="error" role="alert">
            {data.error}
          </p>
        </Card>
      )}
      {data.status === "cancelled" && <p className="muted">Execução cancelada.</p>}
      {data.status === "succeeded" && <RunResultView run={data} />}

      {!active && (
        <div>
          <button
            className="danger"
            onClick={() => remove.mutate(id, { onSuccess: () => navigate("/runs") })}
            disabled={remove.isPending}
          >
            Remover execução
          </button>
        </div>
      )}
    </div>
  );
}
