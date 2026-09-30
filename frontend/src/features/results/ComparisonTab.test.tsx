import { render, screen } from "@testing-library/react";

import type { Comparison } from "../../api/types";
import { ComparisonTab } from "./ComparisonTab";

const base: Comparison = {
  evaluated_pairs: 3,
  reference: { ground_truth_pairs: 1, ground_truth_prevalence: 1 / 3, all_pairs_baseline_f1: 0.5 },
  overlap: [{ strategy: "PCMCI", shared_pairs: 1, only_method: 1, only_ensemble: 0, jaccard: 0.5 }],
  rows: [
    { strategy: "ENSEMBLE_AUTO", returned_edges: 1, detected_pairs: 1, precision: 1, recall: 1, f1_score: 1,
      f1_minus_baseline: 0.5, structural_hamming_distance: 0, average_precision: 1, roc_auc: 1,
      false_positive_pairs: [], false_negative_pairs: [] },
    { strategy: "PCMCI", returned_edges: 2, detected_pairs: 2, precision: 0.5, recall: 1, f1_score: 0.67,
      f1_minus_baseline: 0.17, structural_hamming_distance: 1, average_precision: 0.8, roc_auc: 0.9,
      false_positive_pairs: [["B", "C"]], false_negative_pairs: [] },
  ],
};

describe("ComparisonTab", () => {
  it("tabula as estratégias, destaca o ensemble e lista pares extras", () => {
    render(<ComparisonTab comparison={base} panel={null} />);
    expect(screen.getByRole("img", { name: /Precisão, recall e F1/ })).toBeInTheDocument();
    const ensembleRow = screen.getByRole("cell", { name: "ENSEMBLE_AUTO" }).closest("tr");
    expect(ensembleRow).toHaveClass("highlight");
    expect(screen.getByText("B~C")).toBeInTheDocument();
  });

  it("sem grafo verdadeiro mostra só contagens e sobreposição", () => {
    const rows = base.rows.map(({ strategy, returned_edges, detected_pairs }) => ({ strategy, returned_edges, detected_pairs }));
    render(<ComparisonTab comparison={{ ...base, rows, reference: null }} panel={null} />);
    expect(screen.getByText(/Sem grafo verdadeiro/)).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "F1" })).not.toBeInTheDocument();
    expect(screen.getByText("Sobreposição de cada método com o ensemble (pares não direcionados)")).toBeInTheDocument();
  });

  it("exibe a evidência de painel quando presente", () => {
    render(
      <ComparisonTab
        comparison={base}
        panel={{ trajectory_count: 480, max_lag: 1, context_nodes: 20, error: null,
          ranking_metrics: { roc_auc: 0.9, average_precision: 0.8, random_average_precision: 0.3 },
          top_pairs: [{ source: "A", target: "B", score: 0.7 }] }}
      />,
    );
    expect(screen.getByText(/480 trajetórias independentes/)).toBeInTheDocument();
    expect(screen.getByText("A ~ B")).toBeInTheDocument();
  });
});
