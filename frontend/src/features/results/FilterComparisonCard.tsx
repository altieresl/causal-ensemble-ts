import type { ReactNode } from "react";

import type { FilterComparison, FilterVariantSummary } from "../../api/types";
import { BarChart } from "../../components/charts";
import { Badge, Card, TableWrap } from "../../components/ui";
import { formatNumber } from "../../lib/format";
import { formatElapsed } from "../../lib/useElapsed";

const seconds = (value: number) => formatElapsed(Math.round(value));

function verdict(comparison: FilterComparison): string {
  const { soft, rigid, seconds_saved_by_rigid: saved, speedup_rigid: speedup } = comparison;
  if (rigid.outcome === "insufficient_candidates") {
    return "O filtro rígido não formou ensemble (menos de 2 métodos compatíveis); só o suave produziu resultado.";
  }
  if (comparison.excluded_by_rigid.length === 0) {
    return "Nenhum método violou premissas: os dois filtros avaliaram os mesmos candidatos, e a diferença de tempo é só ruído de medição.";
  }
  const faster = saved >= 0 ? "rígido" : "suave";
  const ratio = speedup && speedup > 0 ? (saved >= 0 ? speedup : 1 / speedup) : null;
  const f1 =
    soft.f1_combination_post_hoc != null && rigid.f1_combination_post_hoc != null
      ? soft.f1_combination_post_hoc === rigid.f1_combination_post_hoc
        ? " com o mesmo F1 pós-hoc"
        : ` com F1 pós-hoc ${formatNumber(rigid.f1_combination_post_hoc, 2)} (rígido) × ${formatNumber(soft.f1_combination_post_hoc, 2)} (suave)`
      : "";
  return `O filtro ${faster} foi ${ratio ? `${formatNumber(ratio, 2)}× ` : ""}mais rápido (${seconds(Math.abs(saved))} de diferença)${f1}.`;
}

const row = (label: string, render: (v: FilterVariantSummary) => ReactNode, comparison: FilterComparison) => (
  <tr key={label}>
    <th scope="row">{label}</th>
    <td>{render(comparison.soft)}</td>
    <td>{render(comparison.rigid)}</td>
  </tr>
);

/** Custo × resultado dos dois filtros de premissas: suave (todos os métodos, com alerta) e rígido (exclui os que violam). */
export function FilterComparisonCard({ comparison }: { comparison: FilterComparison }) {
  return (
    <Card title="Filtro suave × filtro rígido: tempo e resultado">
      <p>
        <strong>{verdict(comparison)}</strong>
      </p>
      <p className="muted small">
        O <strong>suave</strong> mantém todos os métodos e deixa a estabilidade sob bootstrap decidir; o <strong>rígido</strong>{" "}
        exclui de antemão os que violam uma premissa, avaliando menos combinações. As duas variantes rodam em sequência,
        na mesma máquina, para a medida de tempo ser comparável; ainda assim é uma única medição.
      </p>
      <BarChart
        title="Tempo gasto (segundos)"
        max={Math.max(comparison.soft.elapsed_seconds, comparison.rigid.elapsed_seconds, 1)}
        groups={[
          { label: "Filtro suave", values: { t: comparison.soft.elapsed_seconds } },
          { label: "Filtro rígido", values: { t: comparison.rigid.elapsed_seconds } },
        ]}
        series={[{ key: "t", label: "Tempo (s)", className: "bar-0" }]}
      />
      <TableWrap>
        <table>
          <thead>
            <tr>
              <th />
              <th>Suave</th>
              <th>Rígido</th>
            </tr>
          </thead>
          <tbody>
            {row("Tempo gasto", (v) => seconds(v.elapsed_seconds), comparison)}
            {row("Métodos candidatos", (v) => v.candidate_methods.length, comparison)}
            {row("Combinações avaliadas", (v) => v.combinations_evaluated || "—", comparison)}
            {row(
              "Resultado",
              (v) =>
                v.outcome === "completed" ? (
                  <Badge tone="ok">ensemble formado</Badge>
                ) : (
                  <Badge tone="warn">ensemble não formado</Badge>
                ),
              comparison,
            )}
            {row("Melhor combinação", (v) => v.best_combination?.join(" + ") ?? "—", comparison)}
            {row("F1 pós-hoc (combinação)", (v) => formatNumber(v.f1_combination_post_hoc, 2), comparison)}
            {row("Melhor sozinho", (v) => v.best_single_method ?? "—", comparison)}
            {row("F1 pós-hoc (melhor sozinho)", (v) => formatNumber(v.f1_best_single_post_hoc, 2), comparison)}
          </tbody>
        </table>
      </TableWrap>
      <p className="small">
        <strong>Excluídos pelo rígido:</strong> {comparison.excluded_by_rigid.join(", ") || "nenhum"}
      </p>
    </Card>
  );
}
