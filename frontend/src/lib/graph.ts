import type { Edge } from "../api/types";

export interface NodePosition {
  id: string;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  probability: number;
  lags: number[];
}

/** Layout circular determinístico; o primeiro nó fica no topo. */
export function circularLayout(nodes: string[], radius: number, cx: number, cy: number): NodePosition[] {
  const count = Math.max(nodes.length, 1);
  return nodes.map((id, index) => {
    const angle = (2 * Math.PI * index) / count - Math.PI / 2;
    return { id, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
}

/**
 * Reduz as arestas do resultado ao que o grafo desenha: uma seta por par (origem→destino),
 * com a maior probabilidade e todos os lags. Autoarestas ficam fora (aparecem só na tabela).
 * Sem `selectedOnly`, filtra por `minProbability`.
 */
export function collapseEdges(edges: Edge[], opts: { selectedOnly: boolean; minProbability: number }): GraphEdge[] {
  const byPair = new Map<string, GraphEdge>();
  for (const edge of edges) {
    if (edge.source === edge.target) continue;
    const probability = edge.edge_probability ?? 0;
    const keep = opts.selectedOnly ? edge.ensemble_selected === true : probability >= opts.minProbability;
    if (!keep) continue;
    const key = `${edge.source}→${edge.target}`;
    const current = byPair.get(key);
    const lags = edge.lag == null ? [] : [edge.lag];
    if (!current) {
      byPair.set(key, { source: edge.source, target: edge.target, probability, lags });
    } else {
      current.probability = Math.max(current.probability, probability);
      for (const lag of lags) if (!current.lags.includes(lag)) current.lags.push(lag);
    }
  }
  return [...byPair.values()].sort((a, b) => b.probability - a.probability);
}

/** Ponto na borda do círculo do nó, na direção de `to`, para a seta não ficar sob o nó. */
export function trimToNodeBorder(from: NodePosition, to: NodePosition, nodeRadius: number) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  return { x: to.x - (dx / length) * nodeRadius, y: to.y - (dy / length) * nodeRadius };
}
