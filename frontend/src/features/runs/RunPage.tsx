import { useEffect, useRef } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { isTerminal, useDeleteRun, useRun, useRunResult } from "../../api/hooks";
import type {
  AtlasExperimentResult,
  BenchmarkResult,
  ChatResult,
  ReplicatedResult,
  Run,
  RunResult,
} from "../../api/types";
import { ErrorBoundary } from "../../components/ErrorBoundary";
import { useToast } from "../../components/toast";
import {
  Card,
  ConfirmButton,
  ErrorBox,
  PageHeader,
  PageSkeleton,
  ProgressBar,
  Spinner,
  StatusBadge,
} from "../../components/ui";
import { formatDateTime } from "../../lib/format";
import { formatElapsed, useElapsedSeconds } from "../../lib/useElapsed";
import { AtlasExperimentView, ChatResultView } from "../results/AtlasResultViews";
import { BenchmarkResultView } from "../results/BenchmarkResultView";
import { ReplicatedResultView } from "../results/ReplicatedResultView";
import { ResultView } from "../results/ResultView";
import { KIND_LABELS, summarizeParams } from "./kinds";

type StepState = "done" | "current" | "failed" | "todo";

function Timeline({ status }: { status: Run["status"] }) {
  const failed = status === "failed" || status === "cancelled";
  const steps: { label: string; state: StepState }[] = [
    { label: "Na fila", state: status === "queued" ? "current" : "done" },
    { label: "Executando", state: status === "queued" ? "todo" : status === "running" ? "current" : "done" },
    {
      label: failed ? (status === "failed" ? "Falhou" : "Cancelada") : "Concluída",
      state: status === "succeeded" ? "done" : failed ? "failed" : "todo",
    },
  ];
  return (
    <ol className="timeline plain" aria-label="Etapas da execução">
      {steps.map((step, index) => (
        <li key={step.label} className="row" style={{ gap: ".5rem" }}>
          {index > 0 && <span className="sep" aria-hidden />}
          <span className={`step ${step.state}`} aria-current={step.state === "current" ? "step" : undefined}>
            <span className="dot-s" aria-hidden />
            {step.label}
          </span>
        </li>
      ))}
    </ol>
  );
}

function RunResultView({ run }: { run: Run }) {
  const result = useRunResult<unknown>(run.id, true);
  if (result.isPending) return <PageSkeleton label="Carregando resultado…" />;
  if (result.isError) return <ErrorBox error={result.error} />;
  switch (run.kind) {
    case "pipeline":
      return <ResultView result={result.data as RunResult} runId={run.id} />;
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
  const { notify } = useToast();
  const run = useRun(id);
  const remove = useDeleteRun();
  const status = run.data?.status;
  const active = status !== undefined && !isTerminal(status);
  const elapsed = useElapsedSeconds(run.data?.started_at ?? run.data?.created_at ?? null, active);

  // Avisa quando a execução termina enquanto o usuário está na página (ou em outra aba).
  const previous = useRef(status);
  useEffect(() => {
    if (previous.current && !isTerminal(previous.current) && status && isTerminal(status)) {
      if (status === "succeeded") notify("Execução concluída.", "success");
      else if (status === "failed") notify("A execução falhou. Veja o motivo na página.", "error");
    }
    previous.current = status;
  }, [status, notify]);

  if (run.isPending) return <PageSkeleton />;
  if (run.isError) return <ErrorBox error={run.error} />;
  const data = run.data;
  const datasetId = typeof data.params.dataset_id === "string" ? data.params.dataset_id : null;
  const chips = summarizeParams(data);
  const progress = data.progress as { done: number; total: number; message: string } | null;

  return (
    <div className="stack">
      <PageHeader
        title={`${KIND_LABELS[data.kind]} · ${data.id}`}
        crumbs={[
          { label: "Execuções", to: "/runs" },
          ...(datasetId ? [{ label: datasetId, to: `/datasets/${datasetId}` }] : []),
          { label: data.id },
        ]}
        actions={<StatusBadge status={data.status} />}
        lead={
          <>
            Criada {formatDateTime(data.created_at)} · duração{" "}
            {active
              ? formatElapsed(elapsed ?? null)
              : data.started_at && data.finished_at
                ? formatElapsed(Math.round((Date.parse(data.finished_at) - Date.parse(data.started_at)) / 1000))
                : "—"}
          </>
        }
      />
      <Timeline status={data.status} />
      {chips.length > 0 && (
        <p>
          {chips.map((chip) => (
            <span key={chip} className="chip small">
              {chip}
            </span>
          ))}
          {datasetId && <Link to={`/datasets/${datasetId}`}>ver dataset</Link>}
        </p>
      )}

      {active && (
        <Card>
          <Spinner label={data.status === "queued" ? "Na fila…" : "Executando — você pode sair desta página; a execução continua."} />
          <ProgressBar done={progress?.done} total={progress?.total} label="Progresso da execução" />
          {progress ? (
            <p className="muted small">
              {progress.message} · {progress.done}/{progress.total}
            </p>
          ) : (
            <p className="muted small">Esta análise não informa etapas; o tempo decorrido é mostrado acima.</p>
          )}
          <div>
            <ConfirmButton
              onConfirm={() => remove.mutate(id, { onSuccess: () => notify("Execução cancelada.") })}
              confirmLabel="Confirmar cancelamento"
              disabled={remove.isPending}
              title="Interrompe no próximo ponto de progresso; o resultado é descartado."
            >
              Cancelar execução
            </ConfirmButton>
          </div>
        </Card>
      )}
      {data.status === "failed" && (
        <Card title="Falha">
          <p className="alert" role="alert">
            {data.error}
          </p>
          {datasetId && data.kind === "pipeline" && (
            <div>
              <Link to={`/datasets/${datasetId}/new-run`} className="button">
                Ajustar parâmetros e tentar de novo
              </Link>
            </div>
          )}
        </Card>
      )}
      {data.status === "cancelled" && <p className="muted">Execução cancelada.</p>}
      {data.status === "succeeded" && (
        <ErrorBoundary resetKey={data.id}>
          <RunResultView run={data} />
        </ErrorBoundary>
      )}

      {!active && (
        <div>
          <ConfirmButton
            onConfirm={() =>
              remove.mutate(id, {
                onSuccess: () => {
                  notify("Execução removida.");
                  navigate("/runs");
                },
              })
            }
            confirmLabel="Confirmar remoção"
            disabled={remove.isPending}
          >
            Remover execução
          </ConfirmButton>
        </div>
      )}
    </div>
  );
}
