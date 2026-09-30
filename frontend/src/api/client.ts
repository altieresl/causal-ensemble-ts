import type {
  CreateRunRequest,
  DatasetDetails,
  DatasetProfile,
  DatasetSummary,
  Method,
  ProblemDetails,
  Run,
  RunResult,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

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

export const api = {
  listMethods: () => request<Method[]>("/methods"),
  listDatasets: () => request<DatasetSummary[]>("/datasets"),
  getDataset: (id: string) => request<DatasetDetails>(`/datasets/${encodeURIComponent(id)}`),
  uploadDataset: (file: File, dateColumn?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (dateColumn) form.append("date_column", dateColumn);
    return request<DatasetSummary>("/datasets", { method: "POST", body: form });
  },
  deleteDataset: (id: string) => request<void>(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  profileDataset: (id: string, body: { columns?: string[]; declared_causal_sufficiency?: boolean | null }) =>
    request<DatasetProfile>(`/datasets/${encodeURIComponent(id)}/profile`, json("POST", body)),
  listRuns: () => request<Run[]>("/runs"),
  getRun: (id: string) => request<Run>(`/runs/${encodeURIComponent(id)}`),
  getRunResult: (id: string) => request<RunResult>(`/runs/${encodeURIComponent(id)}/result`),
  createRun: (body: CreateRunRequest) => request<Run>("/runs", json("POST", body)),
  deleteRun: (id: string) => request<void>(`/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
