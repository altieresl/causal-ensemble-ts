import type { AtlasExperimentResult, ChatResult } from "../../api/types";
import { Badge, Card, TableWrap } from "../../components/ui";
import { formatNumber } from "../../lib/format";

const metricLine = (metrics?: Record<string, number> | null) =>
  metrics
    ? `F1 ${formatNumber(metrics.f1_score, 2)} · precisão ${formatNumber(metrics.precision, 2)} · recall ${formatNumber(metrics.recall, 2)} · SHD ${metrics.structural_hamming_distance}`
    : "sem grafo verdadeiro: só métricas cegas";

export function AtlasExperimentView({ result }: { result: AtlasExperimentResult }) {
  if (result.outcome === "insufficient_candidates") {
    return (
      <Card title="Ensemble não formado">
        <p>{result.message}</p>
        <p className="muted">
          Um ensemble precisa de ao menos 2 métodos compatíveis com o perfil. Tente outro dataset ou o filtro suave de premissas.
        </p>
      </Card>
    );
  }
  const flagged = Object.entries(result.assumption_flags ?? {});
  return (
    <div className="stack">
      <Card title="Seleção cega ao gabarito">
        <p>Candidatos compatíveis: {result.candidate_methods?.join(", ")}</p>
        <p>
          Melhor combinação: <strong>{result.best_combination_methods?.join(" + ")}</strong> (performance{" "}
          {formatNumber(result.best_combination_performance_score, 4)})
        </p>
        <p>
          Melhor método sozinho: <strong>{result.best_single_method}</strong> (performance{" "}
          {formatNumber(result.best_single_performance_score, 4)})
        </p>
      </Card>
      <Card title="Avaliação pós-hoc (o gabarito só é consultado agora)">
        <p><strong>Combinação:</strong> {metricLine(result.best_combination_metrics_post_hoc)}</p>
        <p><strong>Melhor sozinho:</strong> {metricLine(result.best_single_metrics_post_hoc)}</p>
        {result.ensemble_beats_best_single_f1 != null && (
          <p>
            Ensemble ≥ melhor método sozinho (F1)?{" "}
            <Badge tone={result.ensemble_beats_best_single_f1 ? "ok" : "warn"}>
              {result.ensemble_beats_best_single_f1 ? "sim" : "não"}
            </Badge>
          </p>
        )}
        <p className="muted">
          Um único experimento não é evidência estatística de superioridade; use a validação com réplicas.
        </p>
      </Card>
      <Card title="Recomendação por premissas">
        <TableWrap>
          <table>
            <thead><tr><th>Método</th><th>Decisão</th><th>Motivo</th></tr></thead>
            <tbody>
              {result.recommendations?.map((rec) => (
                <tr key={rec.framework_method_name}>
                  <td>{rec.framework_method_name}</td>
                  <td><Badge tone={rec.included ? "ok" : "warn"}>{rec.included ? "incluir" : "premissa violada"}</Badge></td>
                  <td>{rec.reasons.join(" ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
        {flagged.length > 0 && (
          <>
            <p className="muted">
              Métodos mantidos apesar de violar uma premissa (sobrevivem ou não pela estabilidade sob bootstrap, nunca
              pelo gabarito):
            </p>
            <ul>
              {flagged.map(([name, reasons]) => (
                <li key={name}>
                  <strong>{name}</strong>
                  {result.flagged_methods_in_best_combination?.includes(name) ? " (na melhor combinação)" : " (descartado)"}:{" "}
                  {reasons.join("; ")}
                </li>
              ))}
            </ul>
          </>
        )}
      </Card>
    </div>
  );
}

export function ChatResultView({ result }: { result: ChatResult }) {
  return (
    <div className="stack">
      <Card title="Perfil enviado ao chat">
        <p className="muted">{result.profile_text}</p>
      </Card>
      <Card title="Filtro estatístico × chat local">
        <p>
          <strong>Em comum:</strong> {result.agreement.shared.join(", ") || "—"}
        </p>
        <p>
          <strong>Só no filtro estatístico:</strong> {result.agreement.only_statistical.join(", ") || "—"}
        </p>
        <p>
          <strong>Só no chat:</strong> {result.agreement.only_chat.join(", ") || "—"}
        </p>
        <p className="muted">
          O filtro estatístico é determinístico; o chat pode divergir dele, inclusive de forma inconsistente com as
          próprias premissas fornecidas no prompt.
        </p>
      </Card>
      <Card title="Decisões do chat, por algoritmo">
        <TableWrap>
          <table>
            <thead><tr><th>Método</th><th>Decisão</th><th>Justificativa</th><th>Notas</th></tr></thead>
            <tbody>
              {result.chat.decisions.map((d) => (
                <tr key={d.name}>
                  <td>{d.name}</td>
                  <td><Badge tone={d.include ? "ok" : "warn"}>{d.include ? "incluir" : "excluir"}</Badge></td>
                  <td>{d.reason}</td>
                  <td>
                    {d.retried && "refeito · "}
                    {d.synthesis_corrected && "síntese corrigida em Python"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>
    </div>
  );
}
