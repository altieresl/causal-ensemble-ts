import type { ReplicatedResult } from "../../api/types";
import { StripPlot } from "../../components/charts";
import { Badge, Card, TableWrap } from "../../components/ui";
import { formatNumber } from "../../lib/format";

const METRICS = ["precision", "recall", "f1_score", "average_precision", "roc_auc"] as const;
const num = (value: unknown): number | null => (typeof value === "number" ? value : null);

export function ReplicatedResultView({ result }: { result: ReplicatedResult }) {
  const order = ["ENSEMBLE_AUTO", "ENSEMBLE", ...result.descriptive.map((d) => d.strategy).filter((s) => !s.startsWith("ENSEMBLE")).sort()];
  const descriptive = order
    .map((strategy) => result.descriptive.find((d) => d.strategy === strategy))
    .filter((row) => row !== undefined);
  const precisionByStrategy = order
    .map((strategy) => ({
      label: strategy,
      values: result.metrics
        .filter((row) => row.strategy === strategy)
        .map((row) => num(row.precision))
        .filter((value): value is number => value !== null),
    }))
    .filter((row) => row.values.length > 0);

  return (
    <div className="stack">
      <Card title="Resumo">
        <p>
          {result.completed_replicates} de {result.replicate_ids.length} réplicas concluídas (trajetórias{" "}
          {result.replicate_ids.join(", ")}). Falhas: {result.failures.length}.
        </p>
        <p className="muted">
          Protocolo pareado: Wilcoxon + correção de Holm + IC bootstrap + taxa de vitórias. Sem conclusão textual
          automática — interprete as tabelas.
        </p>
        {result.failures.length > 0 && (
          <TableWrap>
            <table>
              <thead><tr><th>Réplica</th><th>Erro</th></tr></thead>
              <tbody>
                {result.failures.map((f) => (
                  <tr key={f.replicate_id}><td>{f.replicate_id}</td><td>{f.error_type}: {f.error}</td></tr>
                ))}
              </tbody>
            </table>
          </TableWrap>
        )}
      </Card>

      <Card title="Média e desvio-padrão entre réplicas">
        <TableWrap>
          <table>
            <thead>
              <tr>
                <th>Estratégia</th>
                {METRICS.map((m) => <th key={m}>{m}</th>)}
              </tr>
            </thead>
            <tbody>
              {descriptive.map((row) => (
                <tr key={row.strategy}>
                  <td>{row.strategy}</td>
                  {METRICS.map((m) => (
                    <td key={m}>
                      {formatNumber(num(row[`${m}_mean`]), 3)} ± {formatNumber(num(row[`${m}_std`]), 3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
        <StripPlot title="Precisão por estratégia (um ponto por réplica; traço = média)" rows={precisionByStrategy} />
      </Card>

      <Card title="ENSEMBLE_AUTO versus cada baseline (precisão, pareado por réplica)">
        <TableWrap>
          <table>
            <thead>
              <tr>
                <th>Baseline</th><th>Pares</th><th>Ganho médio</th><th>IC 95%</th><th>Vitórias</th>
                <th>Wilcoxon p</th><th>Holm p</th><th>Amostra confirmatória</th><th>Critério</th>
              </tr>
            </thead>
            <tbody>
              {result.comparison.map((row) => (
                <tr key={row.baseline}>
                  <td>{row.baseline}</td>
                  {row.error ? (
                    <td colSpan={8} className="muted">{row.error}</td>
                  ) : (
                    <>
                      <td>{row.paired_trajectories}</td>
                      <td>{formatNumber(row.mean_improvement, 3)}</td>
                      <td>[{formatNumber(row.confidence_interval_low, 3)}; {formatNumber(row.confidence_interval_high, 3)}]</td>
                      <td>{formatNumber(row.win_rate, 2)}</td>
                      <td>{formatNumber(row.wilcoxon_p_value, 4)}</td>
                      <td>{formatNumber(row.holm_p_value, 4)}</td>
                      <td>{row.confirmatory_sample_available ? "sim" : "não"}</td>
                      <td>
                        <Badge tone={row.superiority_criterion_met ? "ok" : "muted"}>
                          {row.superiority_criterion_met ? "atendido" : "não atendido"}
                        </Badge>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>

      <Card title="Subconjuntos escolhidos pela busca automática">
        <TableWrap>
          <table>
            <thead><tr><th>Combinação</th><th>Réplicas</th></tr></thead>
            <tbody>
              {Object.entries(result.selection_counts).map(([combination, count]) => (
                <tr key={combination}><td>{combination}</td><td>{count}</td></tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>
    </div>
  );
}
