import { render, screen } from "@testing-library/react";

import type { FilterComparison, FilterVariantSummary } from "../../api/types";
import { FilterComparisonCard } from "./FilterComparisonCard";

const variant = (over: Partial<FilterVariantSummary>): FilterVariantSummary => ({
  filter: "suave",
  outcome: "completed",
  message: null,
  elapsed_seconds: 30,
  candidate_methods: ["A", "B", "C", "D"],
  combinations_evaluated: 11,
  best_combination: ["A", "B"],
  best_combination_performance_score: 0.7,
  best_single_method: "A",
  f1_combination_post_hoc: 1,
  f1_best_single_post_hoc: 0.8,
  ...over,
});

const comparison = (over: Partial<FilterComparison> = {}): FilterComparison => ({
  soft: variant({}),
  rigid: variant({ filter: "rigido", elapsed_seconds: 20, candidate_methods: ["A", "B", "C"], combinations_evaluated: 4 }),
  seconds_saved_by_rigid: 10,
  speedup_rigid: 1.5,
  excluded_by_rigid: ["D"],
  ...over,
});

describe("FilterComparisonCard", () => {
  it("resume quanto o filtro rígido economizou e se o F1 mudou", () => {
    render(<FilterComparisonCard comparison={comparison()} />);
    expect(screen.getByText(/O filtro rígido foi 1\.50× mais rápido \(10s de diferença\) com o mesmo F1 pós-hoc/)).toBeInTheDocument();
    expect(screen.getByText("Excluídos pelo rígido:").parentElement).toHaveTextContent("D");
    expect(screen.getByRole("row", { name: /Combinações avaliadas/ })).toHaveTextContent("114");
  });

  it("explica quando o rígido não forma ensemble", () => {
    render(
      <FilterComparisonCard
        comparison={comparison({ rigid: variant({ filter: "rigido", outcome: "insufficient_candidates", combinations_evaluated: 0 }) })}
      />,
    );
    expect(screen.getByText(/O filtro rígido não formou ensemble/)).toBeInTheDocument();
  });

  it("sem exclusões, a diferença de tempo é tratada como ruído", () => {
    render(<FilterComparisonCard comparison={comparison({ excluded_by_rigid: [] })} />);
    expect(screen.getByText(/diferença de tempo é só ruído de medição/)).toBeInTheDocument();
  });
});
