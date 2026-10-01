import { useId, useMemo, useState } from "react";

import type { Edge } from "../../api/types";
import { circularLayout, collapseEdges, trimToNodeBorder } from "../../lib/graph";
import { formatNumber } from "../../lib/format";

interface Props {
  nodes: string[];
  edges: Edge[];
  selectedOnly: boolean;
  minProbability: number;
}

const SIZE = 460;
const NODE_RADIUS = 26;

/**
 * Grafo dirigido em SVG: a seta vai de causa (origem) para efeito (destino); espessura = probabilidade.
 * Passar o mouse (ou focar com o teclado) em um nó destaca suas arestas e esmaece o resto.
 */
export function EdgeGraph({ nodes, edges, selectedOnly, minProbability }: Props) {
  const markerId = useId();
  const [focus, setFocus] = useState<string | null>(null);
  const positions = useMemo(
    () => new Map(circularLayout(nodes, SIZE / 2 - NODE_RADIUS - 40, SIZE / 2, SIZE / 2).map((p) => [p.id, p])),
    [nodes],
  );
  const graphEdges = useMemo(
    () => collapseEdges(edges, { selectedOnly, minProbability }),
    [edges, selectedOnly, minProbability],
  );
  const neighbours = useMemo(() => {
    const set = new Set<string>();
    if (focus) for (const e of graphEdges) if (e.source === focus || e.target === focus) set.add(e.source).add(e.target);
    return set;
  }, [focus, graphEdges]);

  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label={`Grafo causal com ${nodes.length} variáveis e ${graphEdges.length} arestas`}
      className="graph"
    >
      <defs>
        <marker id={markerId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>
      {graphEdges.map((edge) => {
        const from = positions.get(edge.source);
        const to = positions.get(edge.target);
        if (!from || !to) return null;
        const start = trimToNodeBorder(to, from, NODE_RADIUS);
        const end = trimToNodeBorder(from, to, NODE_RADIUS + 3);
        // curva leve: pares A→B e B→A não se sobrepõem
        const mx = (start.x + end.x) / 2 + (end.y - start.y) * 0.12;
        const my = (start.y + end.y) / 2 - (end.x - start.x) * 0.12;
        const lagText = edge.lags.length ? ` (lag ${[...edge.lags].sort((a, b) => a - b).join(", ")})` : "";
        const dim = focus !== null && edge.source !== focus && edge.target !== focus;
        return (
          <path
            key={`${edge.source}-${edge.target}`}
            className={`graph-edge${dim ? " dim" : ""}`}
            d={`M${start.x},${start.y} Q${mx},${my} ${end.x},${end.y}`}
            fill="none"
            stroke="var(--accent)"
            strokeOpacity={0.35 + 0.65 * edge.probability}
            strokeWidth={1 + 4 * edge.probability}
            markerEnd={`url(#${markerId})`}
          >
            <title>{`${edge.source} → ${edge.target}${lagText}: p=${formatNumber(edge.probability, 2)}`}</title>
          </path>
        );
      })}
      {nodes.map((id) => {
        const p = positions.get(id)!;
        const active = focus === id;
        const dim = focus !== null && !active && !neighbours.has(id);
        return (
          <g
            key={id}
            tabIndex={0}
            role="button"
            aria-label={`Variável ${id}`}
            onMouseEnter={() => setFocus(id)}
            onMouseLeave={() => setFocus(null)}
            onFocus={() => setFocus(id)}
            onBlur={() => setFocus(null)}
          >
            <circle cx={p.x} cy={p.y} r={NODE_RADIUS} className={`graph-node${active ? " active" : ""}${dim ? " dim" : ""}`} />
            <text x={p.x} y={p.y} textAnchor="middle" dominantBaseline="central" className="graph-label">
              {id.length > 9 ? `${id.slice(0, 8)}…` : id}
              <title>{id}</title>
            </text>
          </g>
        );
      })}
    </svg>
  );
}
