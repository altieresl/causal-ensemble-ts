import type { ReactNode } from "react";

import { ApiError } from "../api/client";
import type { RunStatus } from "../api/types";

export function Card({ title, actions, children }: { title?: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card-header">
          {title && <h2>{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export const Spinner = ({ label = "Carregando…" }: { label?: string }) => (
  <p className="muted" role="status">
    <span className="spinner" aria-hidden /> {label}
  </p>
);

export function ErrorBox({ error }: { error: unknown }) {
  const message = error instanceof ApiError || error instanceof Error ? error.message : "Erro inesperado.";
  return (
    <p className="error" role="alert">
      {message}
    </p>
  );
}

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: "Na fila",
  running: "Executando",
  succeeded: "Concluída",
  failed: "Falhou",
  cancelled: "Cancelada",
};

export const StatusBadge = ({ status }: { status: RunStatus }) => (
  <span className={`badge badge-${status}`}>{STATUS_LABEL[status]}</span>
);

export const Badge = ({ tone, children }: { tone: "ok" | "warn" | "muted"; children: ReactNode }) => (
  <span className={`badge badge-${tone}`}>{children}</span>
);

export const TableWrap = ({ children }: { children: ReactNode }) => <div className="table-wrap">{children}</div>;
