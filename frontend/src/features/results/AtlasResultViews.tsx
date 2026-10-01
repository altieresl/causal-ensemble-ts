import type { AtlasExperimentResult, ChatDecision, ChatResult } from "../../api/types";
import { Badge, Card, Stat, TableWrap } from "../../components/ui";
import { explainDecision } from "../../lib/chatExplain";
import { formatNumber, formatPercent } from "../../lib/format";

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
      <div className="stats">
        <Stat label="Candidatos compatíveis" value={result.candidate_methods?.length ?? 0} hint={result.candidate_methods?.join(", ")} />
        <Stat label="Melhor combinação" value={result.best_combination_methods?.length ?? 0} hint={result.best_combination_methods?.join(" + ")} />
        <Stat label="Melhor método sozinho" value={result.best_single_method ?? "—"} />
        {result.ensemble_beats_best_single_f1 != null && (
          <Stat
            label="Ensemble ≥ sozinho (F1)"
            value={result.ensemble_beats_best_single_f1 ? "sim" : "não"}
            hint="avaliação pós-hoc"
          />
        )}
      </div>
      <Card title="Seleção cega ao gabarito">
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
                  <td className="wrap">{rec.reasons.join(" ")}</td>
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

const yesNo = (value: boolean | null) => (value == null ? "não declarado" : value ? "sim" : "não");

function FractionBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="fraction-row">
      <div className="row between">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="fraction" role="img" aria-label={`${label}: ${formatPercent(value)}`}>
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

/** Um voto do chat, explicado: o que ele leu, quais premissas aplicou e se concorda com o filtro estatístico. */
function DecisionExplanationCard({ decision }: { decision: ChatDecision }) {
  const explanation = explainDecision(decision);
  return (
    <details className={`explain${decision.agrees ? "" : " disagree"}`}>
      <summary>
        <strong>{decision.name}</strong>
        <Badge tone={decision.include ? "ok" : "warn"}>chat: {decision.include ? "incluir" : "excluir"}</Badge>
        <Badge tone={decision.agrees ? "muted" : "danger"}>
          {decision.agrees ? "concorda com o filtro" : "diverge do filtro"}
        </Badge>
        {explanation.violated.length > 0 && (
          <span className="muted small">viola: {explanation.violated.join(", ")}</span>
        )}
      </summary>
      <div className="body">
        <p><strong>Por quê:</strong> {explanation.verdict}</p>
        {decision.reason && <blockquote><strong>Justificativa do chat:</strong> {decision.reason}</blockquote>}

        <div className="stack-sm">
          <h3>Leitura do perfil × premissas do algoritmo</h3>
          <TableWrap>
            <table>
              <thead>
                <tr>
                  <th>Propriedade</th>
                  <th className="num">% das séries (chat)</th>
                  <th>Maioria do dataset?</th>
                  <th>Algoritmo exige?</th>
                  <th>Resultado</th>
                </tr>
              </thead>
              <tbody>
                {explanation.axes.map((axis) => (
                  <tr key={axis.key}>
                    <td>{axis.label}</td>
                    <td className="num">{axis.pct == null ? "—" : `${formatNumber(axis.pct, 0)}%`}</td>
                    <td>{yesNo(axis.datasetMajority)}</td>
                    <td>{yesNo(axis.required)}</td>
                    <td>
                      {axis.violated == null ? (
                        <Badge tone="muted">indeterminado</Badge>
                      ) : axis.violated ? (
                        <Badge tone="danger">premissa violada</Badge>
                      ) : (
                        <Badge tone="ok">sem conflito</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableWrap>
        </div>

        <div className="stack-sm">
          <h3>Filtro estatístico (premissas das fichas do atlas)</h3>
          <p>
            Decisão: <Badge tone={decision.statistical_included ? "ok" : "warn"}>{decision.statistical_included ? "incluir" : "premissa violada"}</Badge>
          </p>
          {decision.statistical_reasons.length > 0 && (
            <ul>
              {decision.statistical_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
        </div>

        {(decision.retried || decision.synthesis_corrected) && (
          <p className="muted small">
            {decision.retried && "A resposta foi pedida de novo porque contradizia fatos verificáveis (percentuais do perfil). "}
            {decision.synthesis_corrected && "O campo de decisão foi recalculado em Python a partir dos booleanos declarados pelo próprio chat."}
          </p>
        )}
      </div>
    </details>
  );
}

export function ChatResultView({ result }: { result: ChatResult }) {
  const { agreement, chat, profile_summary: profile } = result;
  const divergences = agreement.only_statistical.length + agreement.only_chat.length;
  // Divergências primeiro: é onde a explicação mais importa.
  const decisions = [...chat.decisions].sort((a, b) => Number(a.agrees) - Number(b.agrees) || a.name.localeCompare(b.name));
  return (
    <div className="stack">
      <div className="stats">
        <Stat label="Incluídos pelo chat" value={chat.included.length} hint={`de ${chat.decisions.length} algoritmos`} />
        <Stat label="Incluídos pelo filtro" value={result.statistical.included.length} hint="premissas das fichas" />
        <Stat label="Divergências" value={divergences} hint={divergences === 0 ? "chat e filtro coincidem" : "veja os destacados abaixo"} />
      </div>

      <Card title="Perfil do dataset enviado ao chat">
        <div className="form-grid">
          <FractionBar label="Séries estacionárias" value={profile.stationary_fraction} />
          <FractionBar label="Relações aproximadamente lineares" value={profile.linear_fraction} />
          <FractionBar label="Resíduos não gaussianos" value={profile.non_gaussian_fraction} />
        </div>
        <p className="muted small">
          {profile.n_variables} variáveis · {profile.n_timepoints} observações. A maioria é definida por ≥ 50%; o chat nunca recebe o gabarito.
        </p>
      </Card>

      <Card title="Por que o chat votou assim">
        <p className="muted">
          Cada algoritmo é decidido numa chamada independente: o chat precisa reescrever o percentual relevante do perfil,
          declarar quais premissas o algoritmo exige e só então decidir. Abra cada voto para ver o raciocínio e a comparação
          com o filtro estatístico, que é determinístico e pode divergir do chat.
        </p>
        <div className="stack-sm">
          {decisions.map((decision) => (
            <DecisionExplanationCard key={decision.name} decision={decision} />
          ))}
        </div>
      </Card>

      <Card title="Resumo da concordância">
        <p><strong>Em comum:</strong> {agreement.shared.join(", ") || "—"}</p>
        <p><strong>Só no filtro estatístico:</strong> {agreement.only_statistical.join(", ") || "—"}</p>
        <p><strong>Só no chat:</strong> {agreement.only_chat.join(", ") || "—"}</p>
      </Card>
    </div>
  );
}
