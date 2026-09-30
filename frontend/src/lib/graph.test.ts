import type { Edge } from "../api/types";
import { circularLayout, collapseEdges, trimToNodeBorder } from "./graph";

const edge = (over: Partial<Edge>): Edge => ({
  source: "A", target: "B", lag: 1, edge_probability: 0.8, ensemble_score: null, ensemble_selected: true,
  confidence: null, support_ratio: null, dominant_method: null, sign_consensus: null, votes: null, ...over,
});

describe("circularLayout", () => {
  it("coloca o primeiro nó no topo e mantém o raio", () => {
    const [first, second] = circularLayout(["A", "B", "C", "D"], 100, 200, 200);
    expect(first.x).toBeCloseTo(200);
    expect(first.y).toBeCloseTo(100);
    expect(Math.hypot(second.x - 200, second.y - 200)).toBeCloseTo(100);
  });
  it("aceita lista vazia", () => expect(circularLayout([], 10, 0, 0)).toEqual([]));
});

describe("collapseEdges", () => {
  it("agrega lags de um mesmo par mantendo a maior probabilidade", () => {
    const result = collapseEdges(
      [edge({ lag: 1, edge_probability: 0.6 }), edge({ lag: 2, edge_probability: 0.9 })],
      { selectedOnly: true, minProbability: 0 },
    );
    expect(result).toEqual([{ source: "A", target: "B", probability: 0.9, lags: [1, 2] }]);
  });
  it("preserva a direção: A→B e B→A são arestas distintas", () => {
    const result = collapseEdges([edge({}), edge({ source: "B", target: "A" })], { selectedOnly: true, minProbability: 0 });
    expect(result).toHaveLength(2);
  });
  it("ignora autoarestas", () => {
    expect(collapseEdges([edge({ target: "A" })], { selectedOnly: true, minProbability: 0 })).toEqual([]);
  });
  it("filtra por seleção ou por probabilidade mínima", () => {
    const edges = [edge({ ensemble_selected: false, edge_probability: 0.7 })];
    expect(collapseEdges(edges, { selectedOnly: true, minProbability: 0 })).toHaveLength(0);
    expect(collapseEdges(edges, { selectedOnly: false, minProbability: 0.8 })).toHaveLength(0);
    expect(collapseEdges(edges, { selectedOnly: false, minProbability: 0.5 })).toHaveLength(1);
  });
});

describe("trimToNodeBorder", () => {
  it("recua o ponto final pelo raio do nó", () => {
    const point = trimToNodeBorder({ id: "a", x: 0, y: 0 }, { id: "b", x: 100, y: 0 }, 20);
    expect(point).toEqual({ x: 80, y: 0 });
  });
});
