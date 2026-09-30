import {
  OBJECTIVE_LABELS,
  allRelations,
  describeObjective,
  needsPrimary,
  relationKey,
  type Objective,
  type ObjectiveType,
  type Relation,
} from "../../lib/objective";

interface Props {
  nodes: string[];
  objective: Objective;
  onObjectiveChange: (objective: Objective) => void;
  specific: Relation[];
  onSpecificChange: (relations: Relation[]) => void;
  relationCount: number;
}

/** Objetivo de análise: define quais relações origem → destino o pipeline avalia. */
export function RelationSelector({ nodes, objective, onObjectiveChange, specific, onSpecificChange, relationCount }: Props) {
  const set = (patch: Partial<Objective>) => onObjectiveChange({ ...objective, ...patch });
  const chosen = new Set(specific.map(relationKey));
  const toggle = (relation: Relation) =>
    onSpecificChange(
      chosen.has(relationKey(relation))
        ? specific.filter((r) => relationKey(r) !== relationKey(relation))
        : [...specific, relation],
    );

  return (
    <div className="stack">
      <div className="form-grid">
        <label>
          Objetivo
          <select
            value={objective.type}
            onChange={(e) => {
              const type = e.target.value as ObjectiveType;
              set({
                type,
                primary_variable: needsPrimary(type) ? (objective.primary_variable ?? nodes[0] ?? null) : null,
                secondary_variable:
                  type === "compare_directions" ? (objective.secondary_variable ?? nodes[1] ?? null) : null,
              });
            }}
          >
            {(Object.keys(OBJECTIVE_LABELS) as ObjectiveType[]).map((type) => (
              <option key={type} value={type}>
                {OBJECTIVE_LABELS[type]}
              </option>
            ))}
          </select>
        </label>
        {needsPrimary(objective.type) && (
          <label>
            Variável principal
            <select value={objective.primary_variable ?? ""} onChange={(e) => set({ primary_variable: e.target.value })}>
              {nodes.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        )}
        {objective.type === "compare_directions" && (
          <label>
            Segunda variável
            <select value={objective.secondary_variable ?? ""} onChange={(e) => set({ secondary_variable: e.target.value })}>
              {nodes.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="muted">
        {describeObjective(objective)} <strong>{relationCount}</strong> relação(ões) definida(s).
      </p>
      {relationCount === 0 && <p className="error">Selecione ao menos 1 relação para analisar.</p>}

      {objective.type === "specific_relations" && (
        <div className="stack">
          <div className="row">
            <button type="button" onClick={() => onSpecificChange(allRelations(nodes))}>
              Selecionar todas
            </button>
            <button type="button" onClick={() => onSpecificChange([])}>
              Limpar seleção
            </button>
          </div>
          <div className="table-wrap">
            <table aria-label="Relações origem por destino">
              <thead>
                <tr>
                  <th>origem ↓ / destino →</th>
                  {nodes.map((n) => (
                    <th key={n}>{n}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {nodes.map((source) => (
                  <tr key={source}>
                    <th scope="row">{source}</th>
                    {nodes.map((target) => (
                      <td key={target}>
                        {source !== target && (
                          <input
                            type="checkbox"
                            aria-label={`${source} → ${target}`}
                            checked={chosen.has(relationKey([source, target]))}
                            onChange={() => toggle([source, target])}
                          />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
