import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { useCreateRun, useDataset, useMethods } from "../../api/hooks";
import type { ExpertRule } from "../../api/types";
import { Card, ErrorBox, Spinner } from "../../components/ui";
import { allRelations, relationsFor, type Objective, type Relation } from "../../lib/objective";
import { ExpertRulesEditor } from "./ExpertRulesEditor";
import { RelationSelector } from "./RelationSelector";

const toggle = (list: string[], item: string) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);
const optionalInt = (value: string) => (value.trim() === "" ? null : Number(value));

export function NewRunPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const dataset = useDataset(id);
  const methods = useMethods();
  const createRun = useCreateRun();

  const [columns, setColumns] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [maxLag, setMaxLag] = useState(2);
  const [quickMode, setQuickMode] = useState(false);
  const [nBootstrap, setNBootstrap] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [makeStationary, setMakeStationary] = useState(true);
  const [normalize, setNormalize] = useState(true);
  const [decomposition, setDecomposition] = useState("");
  const [trajectory, setTrajectory] = useState("");
  const [panelEvidence, setPanelEvidence] = useState(true);
  const [rules, setRules] = useState<ExpertRule[]>([]);
  const [objective, setObjective] = useState<Objective>({
    type: "full_structure",
    primary_variable: null,
    secondary_variable: null,
  });
  const [specific, setSpecific] = useState<Relation[]>([]);

  useEffect(() => {
    if (dataset.data) {
      setColumns(dataset.data.selected_columns);
      setMaxLag(dataset.data.default_max_lag);
      setDecomposition(dataset.data.decomposition_period ? String(dataset.data.decomposition_period) : "");
    }
  }, [dataset.data]);
  useEffect(() => {
    if (methods.data) setSelectedMethods(methods.data.map((m) => m.name));
  }, [methods.data]);

  // Mantém o objetivo coerente quando o usuário remove variáveis.
  const relations = useMemo(() => relationsFor(objective, columns, specific), [objective, columns, specific]);
  useEffect(() => {
    setObjective((current) => ({
      ...current,
      primary_variable: current.primary_variable && columns.includes(current.primary_variable) ? current.primary_variable : null,
      secondary_variable:
        current.secondary_variable && columns.includes(current.secondary_variable) ? current.secondary_variable : null,
    }));
  }, [columns]);

  if (dataset.isPending || methods.isPending) return <Spinner />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  if (methods.isError) return <ErrorBox error={methods.error} />;

  const details = dataset.data;
  const validRules = rules.filter((r) => r.source && r.target && r.source !== r.target);
  const canSubmit = columns.length >= 2 && selectedMethods.length >= 2 && relations.length > 0 && !createRun.isPending;
  const fullStructure = relations.length === allRelations(columns).length;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    createRun.mutate(
      {
        dataset_id: id,
        columns,
        max_lag: maxLag,
        make_stationary: makeStationary,
        normalize,
        quick_mode: quickMode,
        n_bootstrap: nBootstrap ? Number(nBootstrap) : null,
        methods: selectedMethods,
        expert_knowledge: validRules,
        ensemble_threshold: threshold,
        decomposition_period: optionalInt(decomposition),
        trajectory_index: details.trajectory_count > 1 ? optionalInt(trajectory) : null,
        panel_evidence: panelEvidence,
        selected_relations: fullStructure ? null : relations,
        objective,
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <h1>Nova execução — {details.entry.name}</h1>

      <Card title="Variáveis">
        <div className="checks">
          {details.available_columns.map((c) => (
            <label key={c} className="check">
              <input type="checkbox" checked={columns.includes(c)} onChange={() => setColumns(toggle(columns, c))} />
              {c}
            </label>
          ))}
        </div>
        {columns.length < 2 && <p className="error">Selecione ao menos 2 variáveis.</p>}
      </Card>

      <Card title="Objetivo da análise">
        <RelationSelector
          nodes={columns}
          objective={objective}
          onObjectiveChange={setObjective}
          specific={specific}
          onSpecificChange={setSpecific}
          relationCount={relations.length}
        />
      </Card>

      <Card title="Métodos candidatos">
        <div className="checks">
          {methods.data.map((m) => (
            <label key={m.name} className="check">
              <input
                type="checkbox"
                checked={selectedMethods.includes(m.name)}
                onChange={() => setSelectedMethods(toggle(selectedMethods, m.name))}
              />
              {m.name}
            </label>
          ))}
        </div>
        {selectedMethods.length < 2 && <p className="error">Um ensemble precisa de ao menos 2 métodos.</p>}
      </Card>

      <Card title="Parâmetros">
        <div className="form-grid">
          <label>
            Lag máximo
            <input type="number" min={1} max={20} value={maxLag} onChange={(e) => setMaxLag(Number(e.target.value))} />
          </label>
          <label>
            Bootstraps (vazio = padrão)
            <input type="number" min={1} max={100} value={nBootstrap} onChange={(e) => setNBootstrap(e.target.value)} />
          </label>
          <label>
            Período de decomposição sazonal (vazio = nenhum)
            <input type="number" min={2} value={decomposition} onChange={(e) => setDecomposition(e.target.value)} />
          </label>
          {details.trajectory_count > 1 && (
            <label>
              Trajetória (0–{details.trajectory_count - 1}; vazio = padrão)
              <input
                type="number"
                min={0}
                max={details.trajectory_count - 1}
                value={trajectory}
                onChange={(e) => setTrajectory(e.target.value)}
              />
            </label>
          )}
          <label>
            Limiar do ensemble: {threshold.toFixed(2)}
            <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          </label>
        </div>
        <div className="checks">
          <label className="check">
            <input type="checkbox" checked={quickMode} onChange={(e) => setQuickMode(e.target.checked)} />
            Modo rápido (busca até 3 métodos, menos bootstraps)
          </label>
          <label className="check">
            <input type="checkbox" checked={makeStationary} onChange={(e) => setMakeStationary(e.target.checked)} />
            Tornar estacionário (diferenciação)
          </label>
          <label className="check">
            <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />
            Normalizar
          </label>
          {details.trajectory_count > 1 && details.has_ground_truth && (
            <label className="check">
              <input type="checkbox" checked={panelEvidence} onChange={(e) => setPanelEvidence(e.target.checked)} />
              Evidência de painel com todas as trajetórias (PCMCI; adiciona ~1 min)
            </label>
          )}
        </div>
      </Card>

      <Card title="Conhecimento especialista (opcional)">
        <ExpertRulesEditor columns={columns} rules={rules} onChange={setRules} />
      </Card>

      {createRun.isError && <ErrorBox error={createRun.error} />}
      <div className="row">
        <button type="submit" className="primary" disabled={!canSubmit}>
          {createRun.isPending ? "Enviando…" : "Executar pipeline"}
        </button>
        <span className="muted">A execução pode levar vários minutos; você acompanha o status na próxima tela.</span>
      </div>
    </form>
  );
}
