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

export const useProfile = () =>
  useMutation({
    mutationFn: ({ id, ...body }: { id: string; columns?: string[]; declared_causal_sufficiency?: boolean | null }) =>
      api.profileDataset(id, body),
  });

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: api.listRuns });

/** Async Request-Reply: consulta o status até um estado terminal. */
export const useRun = (id: string) =>
  useQuery({
    queryKey: ["runs", id],
    queryFn: () => api.getRun(id),
    refetchInterval: (query) => (query.state.data && isTerminal(query.state.data.status) ? false : 2000),
  });

export const useRunResult = (id: string, enabled: boolean) =>
  useQuery({ queryKey: ["runs", id, "result"], queryFn: () => api.getRunResult(id), enabled });

export function useCreateRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.createRun,
    onSuccess: () => client.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export function useDeleteRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.deleteRun,
    onSuccess: () => client.invalidateQueries({ queryKey: ["runs"] }),
  });
}
