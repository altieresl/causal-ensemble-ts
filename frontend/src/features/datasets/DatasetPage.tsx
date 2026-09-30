import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { useDataset, useProfile } from "../../api/hooks";
import { Badge, Card, ErrorBox, Spinner, TableWrap } from "../../components/ui";
import { formatNumber, formatPercent } from "../../lib/format";

type Sufficiency = "unknown" | "yes" | "no";
const SUFFICIENCY_VALUE: Record<Sufficiency, boolean | null> = { unknown: null, yes: true, no: false };

export function DatasetPage() {
  const { id = "" } = useParams();
  const dataset = useDataset(id);
  const profile = useProfile();
  const [sufficiency, setSufficiency] = useState<Sufficiency>("unknown");

  if (dataset.isPending) return <Spinner />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  const details = dataset.data;
  const columns = Object.keys(details.preview[0] ?? {});

  return (
    <div className="stack">
      <h1>{details.entry.name}</h1>
      <p className="muted">
        {details.n_rows} linhas · {details.selected_columns.length} variáveis ·{" "}
        {details.has_ground_truth ? "com grafo verdadeiro (usado só na validação pós-hoc)" : "sem grafo verdadeiro"}
      </p>

      <Card title="Pré-visualização">
        <TableWrap>
          <table>
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {details.preview.map((row, i) => (
                <tr key={i}>
                  {columns.map((c) => (
                    <td key={c}>{typeof row[c] === "number" ? formatNumber(row[c] as number, 3) : String(row[c] ?? "—")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>

      <Card
        title="Perfil e métodos recomendados"
        actions={
          <button onClick={() => profile.mutate({ id, declared_causal_sufficiency: SUFFICIENCY_VALUE[sufficiency] })} disabled={profile.isPending}>
            {profile.isPending ? "Calculando…" : "Perfilar dataset"}
          </button>
        }
      >
        <label>
          Suficiência causal (todas as causas comuns foram medidas?){" "}
          <select value={sufficiency} onChange={(e) => setSufficiency(e.target.value as Sufficiency)}>
            <option value="unknown">Não sei (padrão)</option>
            <option value="yes">Sim, declarada por especialista</option>
            <option value="no">Não, há confundidores latentes</option>
          </select>
        </label>
        <p className="muted">Não é verificável só com os dados; informe apenas se um especialista afirmou.</p>
        {profile.isError && <ErrorBox error={profile.error} />}
        {profile.data && (
          <>
            <p>
              {formatPercent(profile.data.stationary_fraction)} estacionárias ·{" "}
              {formatPercent(profile.data.linear_fraction)} lineares ·{" "}
              {formatPercent(profile.data.non_gaussian_fraction)} não gaussianas
            </p>
            <TableWrap>
              <table>
                <thead>
                  <tr>
                    <th>Método</th>
                    <th>Decisão</th>
                    <th>Motivo</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.data.recommendations.map((r) => (
                    <tr key={r.method}>
                      <td>{r.method}</td>
                      <td>
                        <Badge tone={r.included ? "ok" : "warn"}>{r.included ? "incluir" : "premissa violada"}</Badge>
                      </td>
                      <td>{r.reasons.join(" ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          </>
        )}
      </Card>

      <Card title="Análises">
        <div className="row">
          <Link to={`/datasets/${id}/new-run`} className="button primary">
            Pipeline robusto →
          </Link>
          <Link to={`/datasets/${id}/atlas-experiment`} className="button">
            Experimento do atlas
          </Link>
          <Link to={`/datasets/${id}/atlas-chat`} className="button">
            Seleção via chat (Ollama)
          </Link>
          {details.supports_replicates && details.has_ground_truth && (
            <Link to={`/datasets/${id}/validation`} className="button">
              Validação com réplicas
            </Link>
          )}
        </div>
        {details.trajectory_count > 1 && (
          <p className="muted">{details.trajectory_count} trajetórias independentes disponíveis neste dataset.</p>
        )}
      </Card>
    </div>
  );
}
