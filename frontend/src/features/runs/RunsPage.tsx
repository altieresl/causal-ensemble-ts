import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { isTerminal, useRuns } from "../../api/hooks";
import type { Run } from "../../api/types";
import { Card, EmptyState, ErrorBox, PageHeader, PageSkeleton, ProgressBar, StatusBadge, TableWrap } from "../../components/ui";
import { formatDateTime, formatDuration } from "../../lib/format";
import { KIND_LABELS, summarizeParams } from "./kinds";

const STATUS_FILTERS: { value: Run["status"] | "all" | "active"; label: string }[] = [
  { value: "all", label: "Todas" },
  { value: "active", label: "Em andamento" },
  { value: "succeeded", label: "Concluídas" },
  { value: "failed", label: "Com falha" },
  { value: "cancelled", label: "Canceladas" },
];

export function RunsPage() {
  const runs = useRuns();
  const [status, setStatus] = useState<(typeof STATUS_FILTERS)[number]["value"]>("all");
  const [kind, setKind] = useState<Run["kind"] | "all">("all");

  const visible = useMemo(
    () =>
      (runs.data ?? []).filter((run) => {
        if (kind !== "all" && run.kind !== kind) return false;
        if (status === "all") return true;
        if (status === "active") return !isTerminal(run.status);
        return run.status === status;
      }),
    [runs.data, status, kind],
  );

  return (
    <div className="stack">
      <PageHeader
        title="Execuções"
        lead="Histórico das análises. As que estão em andamento se atualizam sozinhas e continuam rodando se você sair da página."
      />
      <div className="row">
        <div className="segmented" role="group" aria-label="Filtrar por status">
          {STATUS_FILTERS.map((filter) => (
            <button key={filter.value} aria-pressed={status === filter.value} onClick={() => setStatus(filter.value)}>
              {filter.label}
            </button>
          ))}
        </div>
        <label className="check">
          Tipo
          <select value={kind} onChange={(e) => setKind(e.target.value as Run["kind"] | "all")}>
            <option value="all">Todos</option>
            {(Object.keys(KIND_LABELS) as Run["kind"][]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
        </label>
      </div>

      {runs.isPending && <PageSkeleton />}
      {runs.isError && <ErrorBox error={runs.error} />}
      {runs.data && visible.length === 0 && (
        <EmptyState title={runs.data.length === 0 ? "Nenhuma execução ainda" : "Nenhuma execução com esse filtro"}>
          {runs.data.length === 0 && (
            <>
              Escolha um <Link to="/">dataset</Link> e inicie uma análise.
            </>
          )}
        </EmptyState>
      )}
      {runs.data && visible.length > 0 && (
        <Card>
          <TableWrap>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Tipo</th>
                  <th>Dataset</th>
                  <th>Status</th>
                  <th>Criada</th>
                  <th>Duração</th>
                  <th>Parâmetros</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => {
                  const progress = run.progress as { done: number; total: number } | null;
                  return (
                    <tr key={run.id}>
                      <td>
                        <Link to={`/runs/${run.id}`}>{run.id}</Link>
                      </td>
                      <td>{KIND_LABELS[run.kind]}</td>
                      <td>{typeof run.params.dataset_id === "string" ? run.params.dataset_id : "—"}</td>
                      <td>
                        <StatusBadge status={run.status} />
                        {!isTerminal(run.status) && progress && (
                          <ProgressBar done={progress.done} total={progress.total} label={`Progresso de ${run.id}`} />
                        )}
                      </td>
                      <td>{formatDateTime(run.created_at)}</td>
                      <td>{formatDuration(run.started_at, run.finished_at)}</td>
                      <td className="wrap muted small">{summarizeParams(run).join(" · ") || "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableWrap>
        </Card>
      )}
    </div>
  );
}
