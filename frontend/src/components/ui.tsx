import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { ApiError } from "../api/client";
import type { RunStatus } from "../api/types";

export function Card({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card${className ? ` ${className}` : ""}`}>
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

export interface Crumb {
  label: string;
  to?: string;
}

/** Título da página + migalhas + descrição; também atualiza o título da aba. */
export function PageHeader({
  title,
  lead,
  crumbs,
  actions,
}: {
  title: string;
  lead?: ReactNode;
  crumbs?: Crumb[];
  actions?: ReactNode;
}) {
  useEffect(() => {
    document.title = `${title} · Causal Discovery TS`;
  }, [title]);
  return (
    <header className="page-header">
      {crumbs && crumbs.length > 0 && (
        <nav className="breadcrumbs" aria-label="Você está em">
          {crumbs.map((crumb, index) => (
            <span key={`${crumb.label}-${index}`} className="row" style={{ gap: ".4rem" }}>
              {index > 0 && <span aria-hidden>›</span>}
              {crumb.to ? <Link to={crumb.to}>{crumb.label}</Link> : <span aria-current="page">{crumb.label}</span>}
            </span>
          ))}
        </nav>
      )}
      <div className="row between">
        <h1>{title}</h1>
        {actions}
      </div>
      {lead && <p className="lead">{lead}</p>}
    </header>
  );
}

export const Spinner = ({ label = "Carregando…" }: { label?: string }) => (
  <p className="muted" role="status">
    <span className="spinner" aria-hidden /> {label}
  </p>
);

export function Skeleton({ height = "1rem", width = "100%" }: { height?: string; width?: string }) {
  return <div className="skeleton" style={{ height, width }} aria-hidden />;
}

/** Carregamento de uma página inteira: evita o "salto" de layout de um spinner solto. */
export function PageSkeleton({ label = "Carregando…" }: { label?: string }) {
  return (
    <div className="stack" role="status" aria-label={label}>
      <Skeleton height="1.8rem" width="40%" />
      <Skeleton height="1rem" width="65%" />
      <Skeleton height="8rem" />
      <Skeleton height="12rem" />
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const message = error instanceof ApiError || error instanceof Error ? error.message : "Erro inesperado.";
  return (
    <p className="alert" role="alert">
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

export const Badge = ({ tone, children }: { tone: "ok" | "warn" | "muted" | "danger"; children: ReactNode }) => (
  <span className={`badge badge-${tone}`}>{children}</span>
);

export const TableWrap = ({ children }: { children: ReactNode }) => <div className="table-wrap">{children}</div>;

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

/** Rótulo + dica + erro associados ao controle (acessível via aria-describedby). */
export function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: ReactNode;
  error?: string | null;
  children: (props: { id: string; "aria-describedby"?: string; "aria-invalid"?: boolean }) => ReactNode;
}) {
  const id = useId();
  const describedBy = [hint ? `${id}-hint` : null, error ? `${id}-error` : null].filter(Boolean).join(" ") || undefined;
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      {children({ id, "aria-describedby": describedBy, "aria-invalid": error ? true : undefined })}
      {hint && (
        <span className="field-hint" id={`${id}-hint`}>
          {hint}
        </span>
      )}
      {error && (
        <span className="field-error" id={`${id}-error`} role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

/** Ação destrutiva em dois cliques (sem `confirm()`): o botão "arma" e volta ao normal após 4 s. */
export function ConfirmButton({
  children,
  confirmLabel = "Confirmar?",
  onConfirm,
  disabled,
  title,
}: {
  children: ReactNode;
  confirmLabel?: string;
  onConfirm: () => void;
  disabled?: boolean;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);
  const timer = useRef<number>();
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <button
      type="button"
      className={`danger${armed ? " armed" : ""}`}
      disabled={disabled}
      title={title}
      onClick={() => {
        if (armed) {
          window.clearTimeout(timer.current);
          setArmed(false);
          onConfirm();
        } else {
          setArmed(true);
          timer.current = window.setTimeout(() => setArmed(false), 4000);
        }
      }}
    >
      {armed ? confirmLabel : children}
    </button>
  );
}

export function ProgressBar({ done, total, label }: { done?: number; total?: number; label: string }) {
  const determinate = typeof done === "number" && typeof total === "number" && total > 0;
  const percent = determinate ? Math.min(100, Math.round(((done as number) / (total as number)) * 100)) : 0;
  return (
    <div
      className={`progress${determinate ? "" : " indeterminate"}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? percent : undefined}
    >
      <span style={determinate ? { width: `${percent}%` } : undefined} />
    </div>
  );
}

/** Barra horizontal inline para valores 0..1 (probabilidades) dentro de tabelas. */
export function ValueBar({ value, digits = 2 }: { value: number | null | undefined; digits?: number }) {
  if (value == null || Number.isNaN(value)) return <span className="muted">—</span>;
  return (
    <span className="bar-cell">
      <span className="bar-track" aria-hidden>
        <span className="bar-fill" style={{ width: `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%` }} />
      </span>
      <span className="num">{value.toFixed(digits)}</span>
    </span>
  );
}
