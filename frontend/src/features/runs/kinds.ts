import type { Run } from "../../api/types";

export const KIND_LABELS: Record<Run["kind"], string> = {
  pipeline: "Pipeline robusto",
  benchmark: "Benchmark sintético",
  replicated_validation: "Validação com réplicas",
  atlas_experiment: "Experimento do atlas",
  atlas_chat: "Seleção via chat",
};

const count = (value: unknown): number | null => (Array.isArray(value) ? value.length : null);

/** Resumo curto dos parâmetros de uma execução, para chips e listas (nunca expõe dados do usuário). */
export function summarizeParams(run: Pick<Run, "kind" | "params">): string[] {
  const p = run.params;
  const chips: string[] = [];
  const add = (label: string, value: unknown) => {
    if (value !== null && value !== undefined && value !== "") chips.push(`${label}: ${String(value)}`);
  };
  switch (run.kind) {
    case "pipeline": {
      add("variáveis", count(p.columns) ?? "padrão");
      add("lag máx.", p.max_lag);
      add("métodos", count(p.methods) ?? "todos");
      if (p.quick_mode) chips.push("modo rápido");
      const relations = count(p.selected_relations);
      const objective = p.objective as { type?: string } | null;
      if (relations !== null) add("relações", relations);
      else if (objective?.type) chips.push("estrutura geral");
      add("trajetória", p.trajectory_index);
      add("sazonalidade", p.decomposition_period);
      break;
    }
    case "benchmark":
      add("amostras", p.n_samples);
      add("ruído ×", p.noise_multiplier);
      add("mudança em", p.index_change);
      add("bootstraps", p.n_bootstrap);
      break;
    case "replicated_validation":
      add("réplicas", p.n_replicates);
      add("em paralelo", p.parallel_replicas ?? "auto");
      add("bootstraps do IC", p.statistical_bootstraps);
      if (p.quick_mode) chips.push("modo rápido");
      break;
    case "atlas_experiment":
      add("bootstraps", p.n_bootstrap);
      add("máx. métodos", p.max_methods);
      add("linhas", p.max_rows ?? "todas");
      break;
    case "atlas_chat":
      add("modelo", p.model ?? "padrão");
      add("chamadas simultâneas", p.parallel_calls);
      break;
  }
  return chips;
}

/** Estimativa grosseira de duração, só para orientar a expectativa (não é promessa). */
export function estimateMinutes(kind: Run["kind"], p: { quickMode?: boolean; replicates?: number; methods?: number }): string {
  switch (kind) {
    case "pipeline":
      if (p.quickMode) return "~1–3 min";
      return (p.methods ?? 8) > 5 ? "~3–10 min" : "~1–4 min";
    case "replicated_validation":
      return p.quickMode ? `~${Math.max(1, Math.round((p.replicates ?? 10) / 4))}–${p.replicates ?? 10} min` : `~${(p.replicates ?? 10) * 2}–${(p.replicates ?? 10) * 5} min`;
    case "benchmark":
      return "~1–5 min";
    default:
      return "~1–3 min";
  }
}
