import { render, screen, within } from "@testing-library/react";

import type { ChatDecision, ChatResult } from "../../api/types";
import { ChatResultView } from "./AtlasResultViews";

const decision = (over: Partial<ChatDecision>): ChatDecision => ({
  name: "PCMCI",
  include: true,
  reason: "premissas satisfeitas",
  retried: false,
  synthesis_corrected: false,
  stationary_fraction_pct: 100,
  dataset_is_majority_stationary: true,
  linear_fraction_pct: 100,
  dataset_is_majority_linear: true,
  non_gaussian_fraction_pct: 0,
  dataset_is_majority_non_gaussian: false,
  algorithm_requires_stationarity: true,
  algorithm_requires_linearity: false,
  algorithm_requires_non_gaussian_errors: false,
  statistical_included: true,
  statistical_reasons: ["Todas as premissas obrigatórias são satisfeitas."],
  agrees: true,
  ...over,
});

const result: ChatResult = {
  profile_text: "perfil",
  profile_summary: { n_variables: 5, n_timepoints: 500, stationary_fraction: 1, linear_fraction: 0.6, non_gaussian_fraction: 0 },
  statistical: { included: ["GES", "PCMCI"], recommendations: [] },
  chat: {
    included: ["PCMCI"],
    excluded: ["GES", "VARLiNGAM"],
    justification: "",
    decisions: [
      decision({ name: "PCMCI" }),
      decision({
        name: "VARLiNGAM",
        include: false,
        statistical_included: false,
        statistical_reasons: ["exige erros não gaussianos"],
        agrees: true,
        algorithm_requires_non_gaussian_errors: true,
      }),
      decision({
        name: "GES",
        include: false,
        reason: "achei que exige linearidade",
        algorithm_requires_linearity: true,
        dataset_is_majority_linear: false,
        linear_fraction_pct: 40,
        agrees: false,
      }),
    ],
  },
  agreement: { only_statistical: ["GES"], only_chat: [], shared: ["PCMCI"] },
};

describe("ChatResultView", () => {
  it("resume incluídos, filtro e divergências", () => {
    render(<ChatResultView result={result} />);
    expect(screen.getByText("Incluídos pelo chat").parentElement).toHaveTextContent("1");
    expect(screen.getByText("Incluídos pelo filtro").parentElement).toHaveTextContent("2");
    expect(screen.getByText("Divergências").parentElement).toHaveTextContent("1");
  });

  it("lista as divergências primeiro e explica o motivo do voto", () => {
    render(<ChatResultView result={result} />);
    const votes = screen.getAllByText(/^(PCMCI|GES|VARLiNGAM)$/, { selector: "summary strong" });
    expect(votes.map((el) => el.textContent)).toEqual(["GES", "PCMCI", "VARLiNGAM"]);

    const ges = votes[0].closest("details")!;
    expect(within(ges).getByText("diverge do filtro")).toBeInTheDocument();
    expect(within(ges).getByText(/exige linearidade e o dataset não a tem/)).toBeInTheDocument();
    expect(within(ges).getByText(/achei que exige linearidade/)).toBeInTheDocument();
    expect(within(ges).getByText("premissa violada")).toBeInTheDocument();
  });

  it("mostra o motivo do filtro estatístico e a concordância", () => {
    render(<ChatResultView result={result} />);
    const varlingam = screen.getByText("VARLiNGAM", { selector: "summary strong" }).closest("details")!;
    expect(within(varlingam).getByText("concorda com o filtro")).toBeInTheDocument();
    expect(within(varlingam).getByText("exige erros não gaussianos")).toBeInTheDocument();
  });
});
