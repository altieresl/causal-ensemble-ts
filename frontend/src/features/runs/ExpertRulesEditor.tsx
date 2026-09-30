import type { Constraint, ExpertRule, Relation } from "../../api/types";

const RELATIONS: Relation[] = ["strong", "weak", "inverse", "none"];
const CONSTRAINTS: Constraint[] = ["soft", "hard"];

interface Props {
  columns: string[];
  rules: ExpertRule[];
  onChange: (rules: ExpertRule[]) => void;
}

export function ExpertRulesEditor({ columns, rules, onChange }: Props) {
  const update = (index: number, patch: Partial<ExpertRule>) =>
    onChange(rules.map((rule, i) => (i === index ? { ...rule, ...patch } : rule)));
  const add = () =>
    onChange([
      ...rules,
      { source: columns[0] ?? "", target: columns[1] ?? "", relation: "strong", confidence: 0.8, constraint: "soft" },
    ]);

  return (
    <div className="stack">
      {rules.length === 0 && <p className="muted">Nenhuma regra. Regras ajustam probabilidades, sem alterar os dados.</p>}
      {rules.map((rule, index) => (
        <div key={index} className="row rule" role="group" aria-label={`Regra ${index + 1}`}>
          <select aria-label="Origem" value={rule.source} onChange={(e) => update(index, { source: e.target.value })}>
            {columns.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <span aria-hidden>→</span>
          <select aria-label="Destino" value={rule.target} onChange={(e) => update(index, { target: e.target.value })}>
            {columns.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <select aria-label="Relação" value={rule.relation} onChange={(e) => update(index, { relation: e.target.value as Relation })}>
            {RELATIONS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
          <select aria-label="Restrição" value={rule.constraint} onChange={(e) => update(index, { constraint: e.target.value as Constraint })}>
            {CONSTRAINTS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <label>
            confiança {rule.confidence.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={rule.confidence}
              onChange={(e) => update(index, { confidence: Number(e.target.value) })}
            />
          </label>
          {rule.source === rule.target && <span className="error">origem e destino devem diferir</span>}
          <button type="button" className="danger" onClick={() => onChange(rules.filter((_, i) => i !== index))}>
            Remover
          </button>
        </div>
      ))}
      <div>
        <button type="button" onClick={add} disabled={columns.length < 2}>
          + Adicionar regra
        </button>
      </div>
    </div>
  );
}
