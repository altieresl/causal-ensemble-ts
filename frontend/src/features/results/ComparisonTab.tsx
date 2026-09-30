import type { Comparison, PanelEvidence } from "../../api/types";
import { BarChart } from "../../components/charts";
import { Card, TableWrap } from "../../components/ui";
import { formatNumber, formatPercent } from "../../lib/format";

const pairs = (list?: string[][]) => (list && list.length ? list.map(([a, b]) => `${a}~${b}`).join(", ") : "—");

/** Ensemble (ENSEMBLE_AUTO) versus cada algoritmo avulso — apenas execução e tabelas, sem conclusão automática. */
export function ComparisonTab({ comparison, panel }: { comparison: Comparison; panel: PanelEvidence | null }) {
  const scored = comparison.rows.some((row) => row.f1_score !== undefined);
  return (
    <div className="stack">
      <Card title="Ensemble versus algoritmos avulsos">
        <p className="muted">
          Direção e lag são colapsados no mesmo par de nós. Para métodos avulsos, toda aresta retornada conta como
          detecção; para o ensemble, apenas as selecionadas. AP e AUROC avaliam o ranking completo dos pares (sem
          limiar). {comparison.evaluated_pairs} pares avaliados.
          {comparison.reference &&
            ` Prevalência estrutural ${formatPercent(comparison.reference.ground_truth_prevalence)}; F1 do baseline (todos os pares): ${formatNumber(comparison.reference.all_pairs_baseline_f1, 2)}.`}
        </p>
        {!scored && <p className="muted">Sem grafo verdadeiro: apenas contagens e sobreposição entre estratégias.</p>}
        {scored && (
          <BarChart
            title="Precisão, recall e F1 por estratégia"
            groups={comparison.rows.map((row) => ({
              label: row.strategy,
              values: { precision: row.precision, recall: row.recall, f1_score: row.f1_score },
            }))}
            series={[
              { key: "precision", label: "Precisão", className: "bar-0" },
              { key: "recall", label: "Recall", className: "bar-1" },
              { key: "f1_score", label: "F1", className: "bar-2" },
            ]}
          />
        )}
        <TableWrap>
          <table>
            <thead>
              <tr>
                <th>Estratégia</th>
                <th>Arestas</th>
                <th>Pares</th>
                {scored && (
                  <>
                    <th>Prec.</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>F1 − baseline</th>
                    <th>SHD</th>
                    <th>AP</th>
                    <th>AUROC</th>
                    <th>Extras (FP)</th>
                    <th>Não recuperados (FN)</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {comparison.rows.map((row) => (
                <tr key={row.strategy} className={row.strategy === "ENSEMBLE_AUTO" ? "highlight" : undefined}>
                  <td>{row.strategy}</td>
                  <td>{row.returned_edges}</td>
                  <td>{row.detected_pairs}</td>
                  {scored && (
                    <>
                      <td>{formatNumber(row.precision, 2)}</td>
                      <td>{formatNumber(row.recall, 2)}</td>
                      <td>{formatNumber(row.f1_score, 2)}</td>
                      <td>{formatNumber(row.f1_minus_baseline, 2)}</td>
                      <td>{row.structural_hamming_distance}</td>
                      <td>{formatNumber(row.average_precision, 2)}</td>
                      <td>{formatNumber(row.roc_auc, 2)}</td>
                      <td>{pairs(row.false_positive_pairs)}</td>
                      <td>{pairs(row.false_negative_pairs)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>

      <Card title="Sobreposição de cada método com o ensemble (pares não direcionados)">
        <TableWrap>
          <table>
            <thead>
              <tr>
                <th>Método</th>
                <th>Em comum</th>
                <th>Só no método</th>
                <th>Só no ensemble</th>
                <th>Jaccard</th>
              </tr>
            </thead>
            <tbody>
              {comparison.overlap.map((row) => (
                <tr key={row.strategy}>
                  <td>{row.strategy}</td>
                  <td>{row.shared_pairs}</td>
                  <td>{row.only_method}</td>
                  <td>{row.only_ensemble}</td>
                  <td>{formatNumber(row.jaccard, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>

      {panel && <PanelCard panel={panel} />}
    </div>
  );
}

function PanelCard({ panel }: { panel: PanelEvidence }) {
  return (
    <Card title="Evidência complementar com todas as trajetórias (PCMCI em painel)">
      <p className="muted">
        {panel.trajectory_count} trajetórias independentes · lag máx. {panel.max_lag} · {panel.context_nodes} nós como
        contexto. Os escores não alteram o ensemble; servem como validação de ranking sem limiar.
      </p>
      {panel.ranking_metrics ? (
        <dl className="metrics">
          <div><dt>AUROC</dt><dd>{formatNumber(panel.ranking_metrics.roc_auc, 3)}</dd></div>
          <div><dt>Average precision</dt><dd>{formatNumber(panel.ranking_metrics.average_precision, 3)}</dd></div>
          <div><dt>AP de um ranking aleatório</dt><dd>{formatNumber(panel.ranking_metrics.random_average_precision, 3)}</dd></div>
        </dl>
      ) : (
        <p className="muted">Ranking não calculado: {panel.error}</p>
      )}
      <TableWrap>
        <table>
          <thead>
            <tr>
              <th>Par</th>
              <th>Escore</th>
            </tr>
          </thead>
          <tbody>
            {panel.top_pairs.map((pair) => (
              <tr key={`${pair.source}-${pair.target}`}>
                <td>{pair.source} ~ {pair.target}</td>
                <td>{formatNumber(pair.score, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrap>
    </Card>
  );
}
