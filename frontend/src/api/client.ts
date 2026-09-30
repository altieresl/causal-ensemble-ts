import type {
  Algorithm,
  AskResponse,
  CreateRunRequest,
  DatasetDetails,
  DatasetProfile,
  DatasetSummary,
  Method,
  ProblemDetails,
  Run,
} from "./types";
import type { components } from "./schema";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

type Schemas = components["schemas"];
export type BenchmarkRequest = Schemas["BenchmarkRequest"];
export type ReplicatedValidationRequest = Schemas["ReplicatedValidationRequest"];
export type AtlasExperimentRequest = Schemas["AtlasExperimentRequest"];
export type AtlasChatRequest = Schemas["AtlasChatRequest"];
export type AskRequest = Schemas["AskRequest"];

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Não foi possível contatar a API.", 0);
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const problem = (await response.json()) as Partial<ProblemDetails>;
      detail = problem.detail ?? problem.title ?? detail;
    } catch {
      /* corpo sem JSON: mantém o status */
    }
    throw new ApiError(detail, response.status);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const id = encodeURIComponent;

export const api = {
  listMethods: () => request<Method[]>("/methods"),
  listDatasets: () => request<DatasetSummary[]>("/datasets"),
  getDataset: (datasetId: string) => request<DatasetDetails>(`/datasets/${id(datasetId)}`),
  uploadDataset: (file: File, dateColumn?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (dateColumn) form.append("date_column", dateColumn);
    return request<DatasetSummary>("/datasets", { method: "POST", body: form });
  },
  deleteDataset: (datasetId: string) => request<void>(`/datasets/${id(datasetId)}`, { method: "DELETE" }),
  profileDataset: (datasetId: string, body: { columns?: string[]; declared_causal_sufficiency?: boolean | null }) =>
    request<DatasetProfile>(`/datasets/${id(datasetId)}/profile`, json("POST", body)),

  listRuns: () => request<Run[]>("/runs"),
  getRun: (runId: string) => request<Run>(`/runs/${id(runId)}`),
  getRunResult: <T>(runId: string) => request<T>(`/runs/${id(runId)}/result`),
  deleteRun: (runId: string) => request<void>(`/runs/${id(runId)}`, { method: "DELETE" }),
  createRun: (body: CreateRunRequest) => request<Run>("/runs", json("POST", body)),
  createBenchmark: (body: BenchmarkRequest) => request<Run>("/runs/benchmark", json("POST", body)),
  createReplicatedValidation: (body: ReplicatedValidationRequest) =>
    request<Run>("/runs/replicated-validation", json("POST", body)),
  createAtlasExperiment: (body: AtlasExperimentRequest) => request<Run>("/runs/atlas-experiment", json("POST", body)),
  createAtlasChat: (body: AtlasChatRequest) => request<Run>("/runs/atlas-chat", json("POST", body)),

  listAlgorithms: () => request<Algorithm[]>("/atlas/algorithms"),
  getAlgorithm: (algorithmId: string) => request<Algorithm>(`/atlas/algorithms/${id(algorithmId)}`),
  ask: (body: AskRequest) => request<AskResponse>("/atlas/ask", json("POST", body)),
};
