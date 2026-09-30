export const formatNumber = (value: number | null | undefined, digits = 3): string =>
  value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);

export const formatPercent = (value: number | null | undefined): string =>
  value == null || Number.isNaN(value) ? "—" : `${Math.round(value * 100)}%`;

export const formatDateTime = (iso: string | null | undefined): string =>
  iso ? new Date(iso).toLocaleString("pt-BR") : "—";

export function formatDuration(start: string | null, end: string | null): string {
  if (!start) return "—";
  const seconds = Math.max(0, Math.round(((end ? Date.parse(end) : Date.now()) - Date.parse(start)) / 1000));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}min ${seconds % 60}s`;
}

/** Cor da célula de um heatmap 0..1 (intensidade do acento). */
export const heatColor = (value: number | null): string =>
  value == null ? "transparent" : `color-mix(in srgb, var(--accent) ${Math.round(value * 85)}%, var(--surface))`;
