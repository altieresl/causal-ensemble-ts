import { useMemo, useState } from "react";

import type { Edge, RunResult } from "../../api/types";
import { Tabs } from "../../components/Tabs";
import { useToast } from "../../components/toast";
import { Card, EmptyState, Stat, TableWrap, ValueBar } from "../../components/ui";
import { downloadText, toCsv } from "../../lib/export";
import { formatNumber, formatPercent, heatColor } from "../../lib/format";
import { collapseEdges } from "../../lib/graph";
import { OBJECTIVE_LABELS, type ObjectiveType } from "../../lib/objective";
import { ComparisonTab } from "./ComparisonTab";
import { EdgeGraph } from "./EdgeGraph";

const TABS = ["Grafo", "Arestas", "Ranking", "Consistência", "Validação", "Comparação"] as const;
type Tab = (typeof TABS)[number];

export function ResultView({ result, runId }: { result: RunResult; runId: string }) {
  const [tab, setTab] = useState<Tab>("Grafo");
  const { notify } = useToast();
  const selected = useMemo(() => collapseEdges(result.edges, { selectedOnly: true, minProbability: 0 }), [result.edges]);
  const objective = result.objective?.type as ObjectiveType | undefined;

  const exportEdges = () => {
    downloadText(`${runId}-arestas.csv`, toCsv(result.edges as unknown as Record<string, unknown>[]), "text/csv");
    notify("Arestas exportadas em CSV.", "success");
  };
  const exportJson = () => {
    downloadText(`${runId}-resultado.json`, JSON.stringify(result, null, 2), "application/json");
    notify("Resultado completo exportado em JSON.", "success");
  };

  return (
    <div className="stack">
      <div className="stats">
        <Stat label="Arestas selecionadas" value={selected.length} hint={`${result.edges.length} candidatas no total`} />
        <Stat label="Variáveis" value={result.columns.length} hint={objective ? OBJECTIVE_LABELS[objective] : undefined} />
        <Stat label="Métodos no ensemble" value={result.best_combination.length} hint={result.best_combination.join(" + ")} />
        {result.validation && (
          <Stat
            label="F1 (validação pós-hoc)"
            value={formatNumber(result.validation.f1_score, 2)}
            hint={`baseline ${formatNumber(result.validation.all_pairs_baseline_f1, 2)}`}
          />
        )}
      </div>

      <Card
        title="Melhor combinação (ENSEMBLE_AUTO)"
        actions={
          <div className="row">
            <button onClick={exportEdges}>Exportar arestas (CSV)</button>
            <button onClick={exportJson}>Exportar tudo (JSON)</button>
          </div>
        }
      >
        <p>
          {result.best_combination.map((m) => (
            <span key={m} className="chip">
              {m} <small className="muted">peso {formatNumber(result.method_weights[m], 2)}</small>
            </span>
          ))}
        </p>
      </Card>

      <Tabs tabs={TABS} value={tab} onChange={setTab} label="Resultados da execução">
        {tab === "Grafo" && <GraphTab result={result} />}
        {tab === "Arestas" && <EdgesTab edges={result.edges} />}
        {tab === "Ranking" && <RankingTab result={result} />}
        {tab === "Consistência" && <ConsistencyTab result={result} />}
        {tab === "Validação" && <ValidationTab result={result} />}
        {tab === "Comparação" &&
          (result.comparison ? (
            <ComparisonTab comparison={result.comparison} panel={result.panel_evidence ?? null} />
          ) : (
            <Card title="Comparação">
              <p className="muted">Esta execução é anterior à comparação; rode novamente para gerá-la.</p>
            </Card>
          ))}
      </Tabs>
    </div>
  );
}

function GraphTab({ result }: { result: RunResult }) {
  const [selectedOnly, setSelectedOnly] = useState(true);
  const [minProbability, setMinProbability] = useState(0.5);
  const listed = useMemo(
    () => collapseEdges(result.edges, { selectedOnly, minProbability }),
    [result.edges, selectedOnly, minProbability],
  );
  return (
    <Card title="Grafo causal">
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={selectedOnly} onChange={(e) => setSelectedOnly(e.target.checked)} />
          Somente arestas selecionadas pelo ensemble
        </label>
        <label className={selectedOnly ? "check muted" : "check"}>
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
      <div className="graph-wrap">
        <EdgeGraph nodes={result.columns} edges={result.edges} selectedOnly={selectedOnly} minProbability={minProbability} />
        <div className="stack-sm">
          <h3>Arestas exibidas ({listed.length})</h3>
          {listed.length === 0 ? (
            <EmptyState title="Nenhuma aresta nesse filtro">
              Reduza a probabilidade mínima ou desmarque “somente selecionadas”.
            </EmptyState>
          ) : (
            <TableWrap>
              <table>
                <thead>
                  <tr>
                    <th>Causa → efeito</th>
                    <th>Lags</th>
                    <th>Prob.</th>
                  </tr>
                </thead>
                <tbody>
                  {listed.map((edge) => (
                    <tr key={`${edge.source}-${edge.target}`}>
                      <td>
                        {edge.source} → {edge.target}
                      </td>
                      <td>{edge.lags.length ? [...edge.lags].sort((a, b) => a - b).join(", ") : "—"}</td>
                      <td>
                        <ValueBar value={edge.probability} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          )}
        </div>
      </div>
      <p className="muted small">
        Passe o mouse sobre uma variável para destacar suas arestas. Setas partem da causa hipotética para o efeito.
        Evidência observacional sugere, mas não prova, causalidade.
      </p>
    </Card>
  );
}

type SortKey = "edge_probability" | "confidence" | "ensemble_score";

function EdgesTab({ edges }: { edges: Edge[] }) {
  const [onlySelected, setOnlySelected] = useState(false);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("edge_probability");
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return edges
      .filter((e) => !onlySelected || e.ensemble_selected)
      .filter((e) => !needle || `${e.source} ${e.target}`.toLowerCase().includes(needle))
      .sort((a, b) => (b[sortKey] ?? -1) - (a[sortKey] ?? -1));
  }, [edges, onlySelected, sortKey, query]);
  return (
    <Card title={`Arestas (${rows.length} de ${edges.length})`}>
      <div className="row">
        <input
          type="search"
          placeholder="Filtrar por variável…"
          aria-label="Filtrar arestas por variável"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="check">
          <input type="checkbox" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
          Somente selecionadas
        </label>
        <label className="check">
          Ordenar por
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
              <th className="num">Lag</th>
              <th>Prob.</th>
              <th>Confiança</th>
              <th className="num">Suporte</th>
              <th>Método dominante</th>
              <th>Selecionada</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={8} className="muted">
                  Nenhuma aresta com esses filtros.
                </td>
              </tr>
            )}
            {rows.map((e, i) => (
              <tr key={`${e.source}-${e.target}-${e.lag}-${i}`}>
                <td>{e.source}</td>
                <td>{e.target}</td>
                <td className="num">{e.lag ?? "—"}</td>
                <td>
                  <ValueBar value={e.edge_probability} />
                </td>
                <td>
                  <ValueBar value={e.confidence} />
                </td>
                <td className="num">{formatPercent(e.support_ratio)}</td>
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
              <th className="num">#</th>
              <th>Combinação</th>
              <th>Performance</th>
              <th>Estabilidade</th>
              <th>Prob. média</th>
              <th>Confiança média</th>
            </tr>
          </thead>
          <tbody>
            {result.ranking.map((r, i) => (
              <tr key={r.combination} className={i === 0 ? "highlight" : undefined}>
                <td className="num">{i + 1}</td>
                <td className="wrap">{r.combination}</td>
                <td>
                  <ValueBar value={r.performance_score} digits={3} />
                </td>
                <td>
                  <ValueBar value={r.mean_stability} digits={3} />
                </td>
                <td className="num">{formatNumber(r.mean_edge_probability)}</td>
                <td className="num">{formatNumber(r.mean_confidence)}</td>
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
                  <td key={labels[j]} className="num" style={{ background: heatColor(value) }}>
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
        <EmptyState title="Sem grafo verdadeiro">
          O dataset não fornece ground truth; não há validação pós-hoc.
        </EmptyState>
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
      <p className="muted small">Direção e lag são ignorados quando o gabarito não os informa.</p>
    </Card>
  );
}
