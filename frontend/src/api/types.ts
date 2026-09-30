// Espelha backend/app/api/schemas.py e o resultado de backend/app/services/pipeline.py.

export type Origin = "builtin" | "upload";
export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type Relation = "strong" | "weak" | "inverse" | "none";
export type Constraint = "soft" | "hard";

export interface DatasetSummary {
  id: string;
  name: string;
  description: string;
  origin: Origin;
  default_max_lag: number;
}

export interface DatasetDetails {
  entry: DatasetSummary;
  available_columns: string[];
  selected_columns: string[];
  n_rows: number;
  preview: Record<string, string | number | null>[];
  has_ground_truth: boolean;
  default_max_lag: number;
}

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

export interface ExpertRule {
  source: string;
  target: string;
  lag?: number | null;
  relation: Relation;
  confidence: number;
  constraint: Constraint;
}

export interface CreateRunRequest {
  dataset_id: string;
  columns?: string[] | null;
  max_lag: number;
  make_stationary: boolean;
  normalize: boolean;
  quick_mode: boolean;
  n_bootstrap?: number | null;
  methods?: string[] | null;
  expert_knowledge: ExpertRule[];
  ensemble_threshold: number;
}

export interface Run {
  id: string;
  status: RunStatus;
  params: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface Method {
  name: string;
  default_weight: number;
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

export interface RunResult {
  columns: string[];
  best_combination: string[];
  edges: Edge[];
  ranking: RankingRow[];
  method_weights: Record<string, number>;
  consistency: { labels: string[]; matrix: (number | null)[][] };
  validation: Validation | null;
}

export interface ProblemDetails {
  title: string;
  status: number;
  detail: string;
}
