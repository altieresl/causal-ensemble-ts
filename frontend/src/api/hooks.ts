import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type { Run } from "./types";

const TERMINAL: Run["status"][] = ["succeeded", "failed", "cancelled"];
export const isTerminal = (status: Run["status"]) => TERMINAL.includes(status);

export const useMethods = () => useQuery({ queryKey: ["methods"], queryFn: api.listMethods, staleTime: Infinity });

export const useDatasets = () => useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });

export const useDataset = (id: string) =>
  useQuery({ queryKey: ["datasets", id], queryFn: () => api.getDataset(id) });

export function useUploadDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ file, dateColumn }: { file: File; dateColumn?: string }) => api.uploadDataset(file, dateColumn),
    onSuccess: () => client.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useDeleteDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: () => client.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

/** Perfil do dataset: roda sozinho (em paralelo com os detalhes) e é cacheado por suficiência declarada. */
export const useProfile = (id: string, declaredCausalSufficiency: boolean | null) =>
  useQuery({
    queryKey: ["profile", id, declaredCausalSufficiency],
    queryFn: () => api.profileDataset(id, { declared_causal_sufficiency: declaredCausalSufficiency }),
    staleTime: Infinity,
    retry: 0,
  });

export const useRuns = () =>
  useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: (query) => (query.state.data?.some((run) => !isTerminal(run.status)) ? 3000 : false),
  });

/** Async Request-Reply: consulta o status até um estado terminal. */
export const useRun = (id: string) =>
  useQuery({
    queryKey: ["runs", id],
    queryFn: () => api.getRun(id),
    refetchInterval: (query) => (query.state.data && isTerminal(query.state.data.status) ? false : 2000),
  });

export const useRunResult = <T>(id: string, enabled: boolean) =>
  useQuery({ queryKey: ["runs", id, "result"], queryFn: () => api.getRunResult<T>(id), enabled });

/** Toda criação de execução (qualquer tipo) invalida a lista de execuções. */
function useCreate<TBody>(create: (body: TBody) => Promise<Run>) {
  const client = useQueryClient();
  return useMutation({ mutationFn: create, onSuccess: () => client.invalidateQueries({ queryKey: ["runs"] }) });
}

export const useCreateRun = () => useCreate(api.createRun);
export const useCreateBenchmark = () => useCreate(api.createBenchmark);
export const useCreateReplicatedValidation = () => useCreate(api.createReplicatedValidation);
export const useCreateAtlasExperiment = () => useCreate(api.createAtlasExperiment);
export const useCreateAtlasChat = () => useCreate(api.createAtlasChat);

export function useDeleteRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.deleteRun,
    onSuccess: () => client.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export const useAlgorithms = () => useQuery({ queryKey: ["atlas", "algorithms"], queryFn: api.listAlgorithms, staleTime: Infinity });

export const useAsk = () => useMutation({ mutationFn: api.ask });
