// Tipos do contrato HTTP derivados do OpenAPI (schema.d.ts, gerado por `npm run generate:api`).
// O perfil e o resultado da execução são dicionários dinâmicos no backend; seus tipos abaixo
// são mantidos à mão (ver backend/app/services/pipeline.py para o formato).
import type { components } from "./schema";

type Schemas = components["schemas"];

export type DatasetSummary = Schemas["DatasetSummary"];
export type DatasetDetails = Schemas["DatasetDetails"];
export type ExpertRule = Schemas["ExpertRule"];
export type CreateRunRequest = Schemas["CreateRunRequest"];
export type Run = Schemas["RunView"];
export type Method = Schemas["MethodView"];
export type Origin = DatasetSummary["origin"];
export type RunStatus = Run["status"];
export type Relation = ExpertRule["relation"];
export type Constraint = ExpertRule["constraint"];

export interface VariableProfile {
  name: string;
  stationary: boolean | null;
  adf_p_value: number | null;
  linear: boolean | null;
  nonlinearity_effect_size: number | null;
  non_gaussian: boolean | null;
}

export interface MethodRecommendation {
  method: string;
  algorithm_id: string;
  included: boolean;
  reasons: string[];
}

export interface DatasetProfile {
  n_variables: number;
  n_timepoints: number;
  stationary_fraction: number;
  linear_fraction: number;
  non_gaussian_fraction: number;
  variables: VariableProfile[];
  recommendations: MethodRecommendation[];
}

export interface Edge {
  source: string;
  target: string;
  lag: number | null;
  edge_probability: number | null;
  ensemble_score: number | null;
  ensemble_selected: boolean | null;
  confidence: number | null;
  support_ratio: number | null;
  dominant_method: string | null;
  sign_consensus: string | number | null;
  votes: number | null;
}

export interface RankingRow {
  combination: string;
  performance_score: number | null;
  mean_stability: number | null;
  mean_edge_probability: number | null;
  mean_confidence: number | null;
}

export interface Validation {
  precision: number;
  recall: number;
  f1_score: number;
  structural_hamming_distance: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  candidate_pairs: number;
  ground_truth_pairs: number;
  all_pairs_baseline_f1: number;
  true_positive_pairs: string[][];
  false_positive_pairs: string[][];
  false_negative_pairs: string[][];
}

export interface ComparisonRow {
  strategy: string;
  returned_edges: number;
  detected_pairs: number;
  // presentes só quando o dataset tem grafo verdadeiro
  precision?: number;
  recall?: number;
  f1_score?: number;
  f1_minus_baseline?: number;
  structural_hamming_distance?: number;
  true_positives?: number;
  false_positives?: number;
  false_negatives?: number;
  average_precision?: number | null;
  roc_auc?: number | null;
  false_positive_pairs?: string[][];
  false_negative_pairs?: string[][];
}

export interface Comparison {
  rows: ComparisonRow[];
  overlap: { strategy: string; shared_pairs: number; only_method: number; only_ensemble: number; jaccard: number | null }[];
  evaluated_pairs: number;
  reference: { ground_truth_pairs: number; ground_truth_prevalence: number; all_pairs_baseline_f1: number } | null;
}

export interface PanelEvidence {
  trajectory_count: number;
  max_lag: number;
  context_nodes: number;
  ranking_metrics: { roc_auc: number; average_precision: number; random_average_precision: number } | null;
  error: string | null;
  top_pairs: { source: string; target: string; score: number }[];
}

export interface RunResult {
  columns: string[];
  objective: { type: string; primary_variable: string | null; secondary_variable: string | null } | null;
  best_combination: string[];
  edges: Edge[];
  ranking: RankingRow[];
  method_weights: Record<string, number>;
  consistency: { labels: string[]; matrix: (number | null)[][] };
  validation: Validation | null;
  comparison: Comparison;
  panel_evidence: PanelEvidence | null;
}

export interface StructuralMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  structural_hamming_distance: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
}

export interface BenchmarkOutcome {
  metrics: StructuralMetrics;
  edges: { source: string; target: string; lag: number; edge_probability: number | null }[];
  true_positives: (string | number)[][];
  false_positives: (string | number)[][];
  false_negatives: (string | number)[][];
  best_combination: string | string[];
}

export interface BenchmarkResult {
  params: Record<string, unknown>;
  ground_truth: { source: string; target: string; lag: number }[];
  clean: BenchmarkOutcome;
  noisy: BenchmarkOutcome;
  delta: Record<string, number>;
}

export interface ReplicatedResult {
  params: Record<string, unknown>;
  columns: string[];
  replicate_ids: number[];
  completed_replicates: number;
  metrics: ({ replicate_id: number; strategy: string } & Record<string, number | string | null>)[];
  selections: { replicate_id: number; auto_combination: string; combinations_evaluated: number }[];
  selection_counts: Record<string, number>;
  failures: { replicate_id: number; error_type: string; error: string }[];
  descriptive: ({ strategy: string } & Record<string, number | string | null>)[];
  comparison: {
    baseline: string;
    paired_trajectories?: number;
    mean_improvement?: number;
    confidence_interval_low?: number;
    confidence_interval_high?: number;
    win_rate?: number;
    wilcoxon_p_value?: number | null;
    holm_p_value?: number | null;
    confirmatory_sample_available?: boolean;
    superiority_criterion_met?: boolean;
    error?: string;
  }[];
}

export interface AtlasExperimentResult {
  outcome: "completed" | "insufficient_candidates";
  message?: string;
  dataset_name?: string;
  n_variables?: number;
  n_timepoints?: number;
  candidate_methods?: string[];
  assumption_flags?: Record<string, string[]>;
  recommendations?: { framework_method_name: string; included: boolean; reasons: string[] }[];
  best_combination_methods?: string[];
  best_combination_performance_score?: number;
  best_single_method?: string;
  best_single_performance_score?: number;
  best_combination_metrics_post_hoc?: Record<string, number> | null;
  best_single_metrics_post_hoc?: Record<string, number> | null;
  all_single_methods_metrics_post_hoc?: Record<string, Record<string, number> | null>;
  single_method_performance_scores?: Record<string, number>;
  ensemble_beats_best_single_f1?: boolean | null;
  flagged_methods_in_best_combination?: string[];
}

export interface ChatDecision {
  name: string;
  include: boolean;
  reason: string;
  retried: boolean;
  synthesis_corrected: boolean;
  // Leitura do perfil e das premissas feita pelo chat (campos obrigatórios do schema JSON dele).
  stationary_fraction_pct: number | null;
  dataset_is_majority_stationary: boolean | null;
  linear_fraction_pct: number | null;
  dataset_is_majority_linear: boolean | null;
  non_gaussian_fraction_pct: number | null;
  dataset_is_majority_non_gaussian: boolean | null;
  algorithm_requires_stationarity: boolean | null;
  algorithm_requires_linearity: boolean | null;
  algorithm_requires_non_gaussian_errors: boolean | null;
  // Comparação com o filtro estatístico determinístico (premissas das fichas do atlas).
  statistical_included: boolean | null;
  statistical_reasons: string[];
  agrees: boolean;
}

export interface ChatResult {
  profile_text: string;
  profile_summary: {
    n_variables: number;
    n_timepoints: number;
    stationary_fraction: number;
    linear_fraction: number;
    non_gaussian_fraction: number;
  };
  statistical: {
    included: string[];
    recommendations: { method: string; included: boolean; reasons: string[] }[];
  };
  chat: {
    included: string[];
    excluded: string[];
    justification: string;
    decisions: ChatDecision[];
  };
  agreement: { only_statistical: string[]; only_chat: string[]; shared: string[] };
}

export interface Algorithm {
  id: string;
  name: string;
  aliases: string[];
  family: string;
  temporal_handling: string;
  output_type: string;
  handles_latent_confounders: boolean;
  handles_nonlinearity: boolean;
  handles_contemporaneous_effects: boolean;
  implemented_in_framework: boolean;
  framework_method_name: string | null;
  verification: string;
  assumptions: { id: string; required: boolean; statement: string }[];
  references: string[];
  sections?: Record<string, string>;
}

export interface AskResponse {
  query: string;
  retrieved: { algorithm_id: string; section: string; text: string; score: number }[];
  answer: string | null;
  error: string | null;
}

export interface ProblemDetails {
  title: string;
  status: number;
  detail: string;
}
