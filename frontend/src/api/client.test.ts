import { ApiError, api } from "./client";

const mockFetch = (response: Partial<Response> & { jsonBody?: unknown }) =>
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true, status: 200, statusText: "OK", json: async () => response.jsonBody, ...response,
    }),
  );

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("devolve o JSON em sucesso e usa o prefixo /api/v1", async () => {
    mockFetch({ jsonBody: [{ name: "PCMCI", default_weight: 1 }] });
    await expect(api.listMethods()).resolves.toEqual([{ name: "PCMCI", default_weight: 1 }]);
    expect(fetch).toHaveBeenCalledWith("/api/v1/methods", undefined);
  });

  it("traduz problem+json em ApiError com o detalhe do servidor", async () => {
    mockFetch({ ok: false, status: 409, statusText: "Conflict", jsonBody: { title: "Conflito", detail: "sem resultado" } });
    const error = await api.getRunResult("run_x").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 409, message: "sem resultado" });
  });

  it("trata 204 sem corpo", async () => {
    mockFetch({ status: 204 });
    await expect(api.deleteRun("run_x")).resolves.toBeUndefined();
  });

  it("falha de rede vira ApiError status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(api.listDatasets()).rejects.toMatchObject({ status: 0 });
  });

  it("escapa ids na URL", async () => {
    mockFetch({ jsonBody: {} });
    await api.getRun("a/b");
    expect(fetch).toHaveBeenCalledWith("/api/v1/runs/a%2Fb", undefined);
  });
});
