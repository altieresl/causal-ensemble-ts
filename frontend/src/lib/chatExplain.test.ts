import type { ChatDecision } from "../api/types";
import { explainDecision } from "./chatExplain";

const base: ChatDecision = {
  name: "PCMCI",
  include: true,
  reason: "ok",
  retried: false,
  synthesis_corrected: false,
  stationary_fraction_pct: 100,
  dataset_is_majority_stationary: true,
  linear_fraction_pct: 80,
  dataset_is_majority_linear: true,
  non_gaussian_fraction_pct: 10,
  dataset_is_majority_non_gaussian: false,
  algorithm_requires_stationarity: true,
  algorithm_requires_linearity: false,
  algorithm_requires_non_gaussian_errors: false,
  statistical_included: true,
  statistical_reasons: [],
  agrees: true,
};

describe("explainDecision", () => {
  it("inclui quando nenhuma premissa exigida é violada", () => {
    const result = explainDecision(base);
    expect(result.violated).toEqual([]);
    expect(result.axes.every((axis) => axis.violated === false)).toBe(true);
    expect(result.verdict).toMatch(/Votou incluir: nenhuma premissa exigida/);
  });

  it("aponta a premissa violada quando o algoritmo exige o que o dataset não tem", () => {
    const result = explainDecision({
      ...base,
      include: false,
      dataset_is_majority_stationary: false,
      stationary_fraction_pct: 20,
    });
    expect(result.violated).toEqual(["estacionariedade"]);
    expect(result.verdict).toMatch(/Votou excluir.*exige estacionariedade/);
  });

  it("junta mais de uma violação e não acusa violação onde o algoritmo não exige", () => {
    const result = explainDecision({
      ...base,
      include: false,
      dataset_is_majority_stationary: false,
      algorithm_requires_non_gaussian_errors: true,
      dataset_is_majority_non_gaussian: false,
    });
    expect(result.violated).toEqual(["estacionariedade", "erros não gaussianos"]);
    expect(result.axes.find((a) => a.key === "linearity")?.violated).toBe(false);
  });

  it("campos não declarados deixam a derivação parcial, sem inventar violação", () => {
    const result = explainDecision({ ...base, algorithm_requires_linearity: null });
    expect(result.axes.find((a) => a.key === "linearity")?.violated).toBeNull();
    expect(result.verdict).toMatch(/parcial/);
  });

  it("registra a correção de síntese feita em Python", () => {
    expect(explainDecision({ ...base, synthesis_corrected: true }).verdict).toMatch(/recalculada em Python/);
  });
});
