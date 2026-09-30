import type { BenchmarkOutcome, BenchmarkResult } from "../../api/types";
import { Card, TableWrap } from "../../components/ui";
import { formatNumber } from "../../lib/format";

const edgeList = (edges: (string | number)[][]) =>
  edges.length ? edges.map(([s, t, lag]) => `${s} → ${t} (lag ${lag})`).join(", ") : "—";

function Outcome({ title, outcome }: { title: string; outcome: BenchmarkOutcome }) {
  const m = outcome.metrics;
  return (
    <Card title={title}>
      <p className="muted">
        Combinação escolhida: {Array.isArray(outcome.best_combination) ? outcome.best_combination.join(" + ") : outcome.best_combination}
      </p>
      <dl className="metrics">
        <div><dt>Precisão</dt><dd>{formatNumber(m.precision, 2)}</dd></div>
        <div><dt>Recall</dt><dd>{formatNumber(m.recall, 2)}</dd></div>
        <div><dt>F1</dt><dd>{formatNumber(m.f1_score, 2)}</dd></div>
        <div><dt>SHD</dt><dd>{m.structural_hamming_distance}</dd></div>
      </dl>
      <p><strong>Verdadeiros positivos:</strong> {edgeList(outcome.true_positives)}</p>
      <p><strong>Falsos positivos:</strong> {edgeList(outcome.false_positives)}</p>
      <p><strong>Falsos negativos:</strong> {edgeList(outcome.false_negatives)}</p>
    </Card>
  );
}

export function BenchmarkResultView({ result }: { result: BenchmarkResult }) {
  const delta = result.delta;
  return (
    <div className="stack">
      <Card title="Estrutura verdadeira (gerador sintético)">
        <TableWrap>
          <table>
            <thead>
              <tr><th>Origem</th><th>Destino</th><th>Lag</th></tr>
            </thead>
            <tbody>
              {result.ground_truth.map((edge) => (
                <tr key={`${edge.source}-${edge.target}-${edge.lag}`}>
                  <td>{edge.source}</td><td>{edge.target}</td><td>{edge.lag}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
        <p className="muted">
          Diferente da análise principal, aqui o gabarito é conhecido por construção (gerador) e só é usado para
          avaliar a saída depois da seleção.
        </p>
      </Card>
      <Outcome title="Série limpa" outcome={result.clean} />
      <Outcome title="Série com mudança no regime de ruído" outcome={result.noisy} />
      <Card title="Robustez">
        <p>
          Variação de F1 com ruído severo: <strong>{formatNumber(delta.f1_score, 2)}</strong> · variação do SHD:{" "}
          <strong>{formatNumber(delta.structural_hamming_distance, 0)}</strong>
        </p>
      </Card>
    </div>
  );
}
