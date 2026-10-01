import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { useDataset, useProfile } from "../../api/hooks";
import { Badge, Card, ErrorBox, PageHeader, PageSkeleton, Skeleton, Stat, TableWrap } from "../../components/ui";
import { formatNumber, formatPercent } from "../../lib/format";

type Sufficiency = "unknown" | "yes" | "no";
const SUFFICIENCY_VALUE: Record<Sufficiency, boolean | null> = { unknown: null, yes: true, no: false };

function Fraction({ label, value, hint }: { label: string; value: number; hint: string }) {
  return (
    <div className="fraction-row">
      <div className="row between">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="fraction" role="img" aria-label={`${label}: ${formatPercent(value)}`}>
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="muted small">{hint}</span>
    </div>
  );
}

const triState = (value: boolean | null, yes: string, no: string) =>
  value === null ? <Badge tone="muted">não testável</Badge> : value ? <Badge tone="ok">{yes}</Badge> : <Badge tone="warn">{no}</Badge>;

export function DatasetPage() {
  const { id = "" } = useParams();
  const [sufficiency, setSufficiency] = useState<Sufficiency>("unknown");
  // Detalhes e perfil são pedidos ao mesmo tempo (queries independentes).
  const dataset = useDataset(id);
  const profile = useProfile(id, SUFFICIENCY_VALUE[sufficiency]);

  if (dataset.isPending) return <PageSkeleton />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  const details = dataset.data;
  const columns = Object.keys(details.preview[0] ?? {});
  const included = profile.data?.recommendations.filter((r) => r.included).length ?? 0;

  return (
    <div className="stack">
      <PageHeader
        title={details.entry.name}
        crumbs={[{ label: "Datasets", to: "/" }, { label: details.entry.name }]}
        lead={details.entry.description}
      />

      <div className="stats">
        <Stat label="Observações" value={details.n_rows} />
        <Stat label="Variáveis" value={details.selected_columns.length} hint={`${details.available_columns.length} disponíveis`} />
        <Stat
          label="Grafo verdadeiro"
          value={details.has_ground_truth ? "sim" : "não"}
          hint={details.has_ground_truth ? "usado só na validação pós-hoc" : undefined}
        />
        {details.trajectory_count > 1 && <Stat label="Trajetórias" value={details.trajectory_count} hint="independentes" />}
      </div>

      <Card title="Análises">
        <div className="grid">
          <Link to={`/datasets/${id}/new-run`} className="card flat interactive" style={{ textDecoration: "none", color: "inherit" }}>
            <h3>Pipeline robusto →</h3>
            <p className="muted small">Ensemble automático com estabilidade por bootstrap, objetivos de análise e conhecimento especialista.</p>
          </Link>
          <Link to={`/datasets/${id}/atlas-experiment`} className="card flat interactive" style={{ textDecoration: "none", color: "inherit" }}>
            <h3>Experimento do atlas</h3>
            <p className="muted small">Seleciona métodos pelas premissas e compara a melhor combinação com o melhor método sozinho.</p>
          </Link>
          <Link to={`/datasets/${id}/atlas-chat`} className="card flat interactive" style={{ textDecoration: "none", color: "inherit" }}>
            <h3>Seleção via chat (Ollama)</h3>
            <p className="muted small">Um LLM local vota em cada algoritmo e explica o motivo; compara com o filtro estatístico.</p>
          </Link>
          {details.supports_replicates && details.has_ground_truth && (
            <Link to={`/datasets/${id}/validation`} className="card flat interactive" style={{ textDecoration: "none", color: "inherit" }}>
              <h3>Validação com réplicas</h3>
              <p className="muted small">Teste estatístico pareado (Wilcoxon + Holm) em trajetórias independentes.</p>
            </Link>
          )}
        </div>
      </Card>

      <Card
        title="Perfil e métodos recomendados"
        actions={
          <label className="check">
            Suficiência causal
            <select value={sufficiency} onChange={(e) => setSufficiency(e.target.value as Sufficiency)}>
              <option value="unknown">Não sei (padrão)</option>
              <option value="yes">Sim, declarada por especialista</option>
              <option value="no">Não, há confundidores latentes</option>
            </select>
          </label>
        }
      >
        <p className="muted small">
          Suficiência causal (todas as causas comuns foram medidas?) não é verificável só com os dados; informe apenas se
          um especialista afirmou.
        </p>
        {profile.isPending && (
          <div className="stack-sm" role="status" aria-label="Calculando perfil">
            <Skeleton height="1rem" width="55%" />
            <Skeleton height="1rem" width="45%" />
            <Skeleton height="6rem" />
          </div>
        )}
        {profile.isError && <ErrorBox error={profile.error} />}
        {profile.data && (
          <>
            <div className="form-grid">
              <Fraction label="Estacionárias" value={profile.data.stationary_fraction} hint="teste ADF, α = 0,05" />
              <Fraction label="Aproximadamente lineares" value={profile.data.linear_fraction} hint="erro fora da amostra: linear × não linear" />
              <Fraction label="Resíduos não gaussianos" value={profile.data.non_gaussian_fraction} hint="assimetria/curtose dos resíduos de um VAR(1)" />
            </div>
            <p>
              <strong>{included}</strong> de {profile.data.recommendations.length} métodos compatíveis com o perfil.
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
                      <td className="wrap">{r.reasons.join(" ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
            <details className="advanced">
              <summary>Perfil por variável</summary>
              <TableWrap>
                <table>
                  <thead>
                    <tr>
                      <th>Variável</th>
                      <th>Estacionária</th>
                      <th className="num">p (ADF)</th>
                      <th>Linear</th>
                      <th>Não gaussiana</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile.data.variables.map((v) => (
                      <tr key={v.name}>
                        <td>{v.name}</td>
                        <td>{triState(v.stationary, "sim", "não")}</td>
                        <td className="num">{formatNumber(v.adf_p_value, 3)}</td>
                        <td>{triState(v.linear, "sim", "não")}</td>
                        <td>{triState(v.non_gaussian, "sim", "não")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableWrap>
            </details>
          </>
        )}
      </Card>

      <Card title="Pré-visualização">
        <TableWrap>
          <table>
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c} className={typeof details.preview[0]?.[c] === "number" ? "num" : undefined}>
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {details.preview.map((row, i) => (
                <tr key={i}>
                  {columns.map((c) => (
                    <td key={c} className={typeof row[c] === "number" ? "num" : undefined}>
                      {typeof row[c] === "number" ? (Number.isInteger(row[c]) ? String(row[c]) : formatNumber(row[c] as number, 3)) : String(row[c] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>
    </div>
  );
}
