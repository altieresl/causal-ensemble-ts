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
