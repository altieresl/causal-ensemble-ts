import type { Run } from "../../api/types";

export const KIND_LABELS: Record<Run["kind"], string> = {
  pipeline: "Pipeline robusto",
  benchmark: "Benchmark sintético",
  replicated_validation: "Validação com réplicas",
  atlas_experiment: "Experimento do atlas",
  atlas_chat: "Seleção via chat",
};
