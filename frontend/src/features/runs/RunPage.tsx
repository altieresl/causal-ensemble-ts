import { Link, useNavigate, useParams } from "react-router-dom";

import { isTerminal, useDeleteRun, useRun, useRunResult } from "../../api/hooks";
import { Card, ErrorBox, Spinner, StatusBadge } from "../../components/ui";
import { formatDateTime, formatDuration } from "../../lib/format";
import { ResultView } from "../results/ResultView";

export function RunPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const run = useRun(id);
  const succeeded = run.data?.status === "succeeded";
  const result = useRunResult(id, succeeded);
  const remove = useDeleteRun();

  if (run.isPending) return <Spinner />;
  if (run.isError) return <ErrorBox error={run.error} />;
  const data = run.data;
  const active = !isTerminal(data.status);

  return (
    <div className="stack">
      <h1>
        Execução {data.id} <StatusBadge status={data.status} />
      </h1>
      <p className="muted">
        Dataset <Link to={`/datasets/${String(data.params.dataset_id)}`}>{String(data.params.dataset_id)}</Link> · criada{" "}
        {formatDateTime(data.created_at)} · duração {formatDuration(data.started_at, data.finished_at)}
      </p>

      {active && (
        <Card>
          <Spinner label={data.status === "queued" ? "Na fila…" : "Executando o pipeline (pode levar minutos)…"} />
          <button
            className="danger"
            onClick={() => remove.mutate(id)}
            disabled={remove.isPending}
            title="O processamento em curso não é interrompido, mas o resultado é descartado."
          >
            Cancelar
          </button>
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
      {succeeded && result.isPending && <Spinner label="Carregando resultado…" />}
      {result.isError && <ErrorBox error={result.error} />}
      {result.data && <ResultView result={result.data} />}

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
