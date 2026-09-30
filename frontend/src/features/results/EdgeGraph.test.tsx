import { render, screen } from "@testing-library/react";

import type { Edge } from "../../api/types";
import { EdgeGraph } from "./EdgeGraph";

const edge = (over: Partial<Edge>): Edge => ({
  source: "A", target: "B", lag: 1, edge_probability: 0.9, ensemble_score: null, ensemble_selected: true,
  confidence: null, support_ratio: null, dominant_method: null, sign_consensus: null, votes: null, ...over,
});

describe("EdgeGraph", () => {
  it("desenha um nó por variável e uma seta por aresta selecionada", () => {
    const { container } = render(
      <EdgeGraph nodes={["A", "B", "C"]} edges={[edge({}), edge({ target: "C", ensemble_selected: false })]} selectedOnly minProbability={0} />,
    );
    expect(container.querySelectorAll("circle")).toHaveLength(3);
    expect(container.querySelectorAll("path[marker-end]")).toHaveLength(1);
    expect(screen.getByRole("img")).toHaveAccessibleName(/3 variáveis e 1 arestas/);
    expect(screen.getByText(/A → B \(lag 1\): p=0.90/)).toBeInTheDocument();
  });

  it("renderiza sem arestas", () => {
    const { container } = render(<EdgeGraph nodes={["A", "B"]} edges={[]} selectedOnly minProbability={0} />);
    expect(container.querySelectorAll("path[marker-end]")).toHaveLength(0);
  });
});
