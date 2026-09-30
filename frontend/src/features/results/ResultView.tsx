import { useMemo, useState } from "react";

import type { Edge, RunResult } from "../../api/types";
import { Card, TableWrap } from "../../components/ui";
import { formatNumber, formatPercent, heatColor } from "../../lib/format";
import { EdgeGraph } from "./EdgeGraph";

const TABS = ["Grafo", "Arestas", "Ranking", "Consistência", "Validação"] as const;
type Tab = (typeof TABS)[number];

export function ResultView({ result }: { result: RunResult }) {
  const [tab, setTab] = useState<Tab>("Grafo");
  return (
    <div className="stack">
      <Card title="Melhor combinação (ENSEMBLE_AUTO)">
        <p>
          {result.best_combination.map((m) => (
            <span key={m} className="chip">
              {m} <small>peso {formatNumber(result.method_weights[m], 2)}</small>
            </span>
          ))}
        </p>
      </Card>
      <div role="tablist" className="tabs">
        {TABS.map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={tab === name}
            className={tab === name ? "tab active" : "tab"}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>
      {tab === "Grafo" && <GraphTab result={result} />}
      {tab === "Arestas" && <EdgesTab edges={result.edges} />}
      {tab === "Ranking" && <RankingTab result={result} />}
      {tab === "Consistência" && <ConsistencyTab result={result} />}
      {tab === "Validação" && <ValidationTab result={result} />}
    </div>
  );
}

function GraphTab({ result }: { result: RunResult }) {
  const [selectedOnly, setSelectedOnly] = useState(true);
  const [minProbability, setMinProbability] = useState(0.5);
  return (
    <Card title="Grafo causal">
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={selectedOnly} onChange={(e) => setSelectedOnly(e.target.checked)} />
          Somente arestas selecionadas pelo ensemble
        </label>
        <label className={selectedOnly ? "disabled" : ""}>
          Probabilidade mínima: {formatNumber(minProbability, 2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={minProbability}
            disabled={selectedOnly}
            onChange={(e) => setMinProbability(Number(e.target.value))}
          />
        </label>
      </div>
      <EdgeGraph nodes={result.columns} edges={result.edges} selectedOnly={selectedOnly} minProbability={minProbability} />
      <p className="muted">
        Setas partem da causa hipotética para o efeito. Evidência observacional sugere, mas não prova, causalidade.
      </p>
    </Card>
  );
}

type SortKey = "edge_probability" | "confidence" | "ensemble_score";

function EdgesTab({ edges }: { edges: Edge[] }) {
  const [onlySelected, setOnlySelected] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("edge_probability");
  const rows = useMemo(
    () =>
      edges
        .filter((e) => !onlySelected || e.ensemble_selected)
        .sort((a, b) => (b[sortKey] ?? -1) - (a[sortKey] ?? -1)),
    [edges, onlySelected, sortKey],
  );
  return (
    <Card title={`Arestas (${rows.length})`}>
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
          Somente selecionadas
        </label>
        <label>
          Ordenar por{" "}
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
            <option value="edge_probability">probabilidade</option>
            <option value="confidence">confiança</option>
            <option value="ensemble_score">score do ensemble</option>
          </select>
        </label>
      </div>
      <TableWrap>
        <table>
          <thead>
            <tr>
              <th>Origem</th>
              <th>Destino</th>
              <th>Lag</th>
              <th>Prob.</th>
              <th>Confiança</th>
              <th>Suporte</th>
              <th>Método dominante</th>
              <th>Selecionada</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={8} className="muted">
                  Nenhuma aresta.
                </td>
              </tr>
            )}
            {rows.map((e, i) => (
              <tr key={`${e.source}-${e.target}-${e.lag}-${i}`}>
                <td>{e.source}</td>
                <td>{e.target}</td>
                <td>{e.lag ?? "—"}</td>
                <td>{formatNumber(e.edge_probability)}</td>
                <td>{formatNumber(e.confidence)}</td>
                <td>{formatPercent(e.support_ratio)}</td>
                <td>{e.dominant_method ?? "—"}</td>
                <td>{e.ensemble_selected ? "sim" : "não"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrap>
    </Card>
  );
}

function RankingTab({ result }: { result: RunResult }) {
  return (
    <Card title="Ranking de combinações de métodos">
      <TableWrap>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Combinação</th>
              <th>Performance</th>
              <th>Estabilidade</th>
              <th>Prob. média</th>
              <th>Confiança média</th>
            </tr>
          </thead>
          <tbody>
            {result.ranking.map((r, i) => (
              <tr key={r.combination}>
                <td>{i + 1}</td>
                <td>{r.combination}</td>
                <td>{formatNumber(r.performance_score)}</td>
                <td>{formatNumber(r.mean_stability)}</td>
                <td>{formatNumber(r.mean_edge_probability)}</td>
                <td>{formatNumber(r.mean_confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrap>
    </Card>
  );
}

function ConsistencyTab({ result }: { result: RunResult }) {
  const { labels, matrix } = result.consistency;
  return (
    <Card title="Consistência entre métodos (Jaccard das arestas)">
      <TableWrap>
        <table>
          <thead>
            <tr>
              <th />
              {labels.map((l) => (
                <th key={l}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((row, i) => (
              <tr key={row}>
                <th scope="row">{row}</th>
                {matrix[i].map((value, j) => (
                  <td key={labels[j]} style={{ background: heatColor(value) }}>
                    {formatNumber(value, 2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrap>
    </Card>
  );
}

const pairs = (list: string[][]) => (list.length ? list.map(([a, b]) => `${a} – ${b}`).join(", ") : "—");

function ValidationTab({ result }: { result: RunResult }) {
  const v = result.validation;
  if (!v) {
    return (
      <Card title="Validação estrutural">
        <p className="muted">O dataset não fornece grafo verdadeiro (ground truth); não há validação pós-hoc.</p>
      </Card>
    );
  }
  return (
    <Card title="Validação estrutural pós-hoc (esqueleto não direcionado)">
      <p className="muted">
        Calculada somente depois da seleção; o gabarito nunca influencia a escolha dos métodos.
      </p>
      <dl className="metrics">
        <div><dt>Precisão</dt><dd>{formatNumber(v.precision, 2)}</dd></div>
        <div><dt>Recall</dt><dd>{formatNumber(v.recall, 2)}</dd></div>
        <div><dt>F1</dt><dd>{formatNumber(v.f1_score, 2)}</dd></div>
        <div><dt>F1 (baseline: todos os pares)</dt><dd>{formatNumber(v.all_pairs_baseline_f1, 2)}</dd></div>
        <div><dt>SHD</dt><dd>{v.structural_hamming_distance}</dd></div>
        <div><dt>Pares possíveis / verdadeiros</dt><dd>{v.candidate_pairs} / {v.ground_truth_pairs}</dd></div>
      </dl>
      <p><strong>Corretos:</strong> {pairs(v.true_positive_pairs)}</p>
      <p><strong>Extras:</strong> {pairs(v.false_positive_pairs)}</p>
      <p><strong>Não recuperados:</strong> {pairs(v.false_negative_pairs)}</p>
      <p className="muted">Direção e lag são ignorados quando o gabarito não os informa.</p>
    </Card>
  );
}
