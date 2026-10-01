import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { useCreateRun, useDataset, useMethods } from "../../api/hooks";
import type { ExpertRule } from "../../api/types";
import { useToast } from "../../components/toast";
import { Card, ErrorBox, Field, PageHeader, PageSkeleton } from "../../components/ui";
import { allRelations, relationsFor, type Objective, type Relation } from "../../lib/objective";
import { ExpertRulesEditor } from "./ExpertRulesEditor";
import { estimateMinutes } from "./kinds";
import { RelationSelector } from "./RelationSelector";

const toggle = (list: string[], item: string) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);
const optionalInt = (value: string) => (value.trim() === "" ? null : Number(value));

const PRESETS = {
  rapido: { label: "Rápido", quick: true, bootstrap: "4", hint: "Busca até 3 métodos; bom para explorar." },
  equilibrado: { label: "Equilibrado", quick: false, bootstrap: "", hint: "Padrão do notebook: todas as combinações." },
  completo: { label: "Completo", quick: false, bootstrap: "20", hint: "Mais bootstraps: estabilidade mais precisa e mais lenta." },
} as const;
type PresetKey = keyof typeof PRESETS;

function Step({ n, title, hint, children }: { n: number; title: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <Card
      title={
        <span className="step-title">
          <span className="step-number" aria-hidden>
            {n}
          </span>
          {title}
        </span>
      }
    >
      {hint && <p className="muted small">{hint}</p>}
      {children}
    </Card>
  );
}

export function NewRunPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const { notify } = useToast();
  const dataset = useDataset(id);
  const methods = useMethods();
  const createRun = useCreateRun();

  const [columns, setColumns] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [maxLag, setMaxLag] = useState(2);
  const [preset, setPreset] = useState<PresetKey | "custom">("equilibrado");
  const [quickMode, setQuickMode] = useState(false);
  const [nBootstrap, setNBootstrap] = useState("");
  const [parallelJobs, setParallelJobs] = useState("");
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

  if (dataset.isPending || methods.isPending) return <PageSkeleton />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  if (methods.isError) return <ErrorBox error={methods.error} />;

  const details = dataset.data;
  const allMethodNames = methods.data.map((m) => m.name);
  const validRules = rules.filter((r) => r.source && r.target && r.source !== r.target);
  const problems = [
    columns.length < 2 ? "Selecione ao menos 2 variáveis." : null,
    selectedMethods.length < 2 ? "Selecione ao menos 2 métodos." : null,
    relations.length === 0 ? "Defina ao menos 1 relação no objetivo da análise." : null,
  ].filter((p): p is string => p !== null);
  const fullStructure = relations.length === allRelations(columns).length;

  const applyPreset = (key: PresetKey) => {
    setPreset(key);
    setQuickMode(PRESETS[key].quick);
    setNBootstrap(PRESETS[key].bootstrap);
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (problems.length > 0) return;
    createRun.mutate(
      {
        dataset_id: id,
        columns,
        max_lag: maxLag,
        make_stationary: makeStationary,
        normalize,
        quick_mode: quickMode,
        n_bootstrap: nBootstrap ? Number(nBootstrap) : null,
        parallel_jobs: parallelJobs ? Number(parallelJobs) : null,
        methods: selectedMethods,
        expert_knowledge: validRules,
        ensemble_threshold: threshold,
        decomposition_period: optionalInt(decomposition),
        trajectory_index: details.trajectory_count > 1 ? optionalInt(trajectory) : null,
        panel_evidence: panelEvidence,
        selected_relations: fullStructure ? null : relations,
        objective,
      },
      {
        onSuccess: (run) => {
          notify("Execução criada. Você acompanha o andamento aqui.", "success");
          navigate(`/runs/${run.id}`);
        },
      },
    );
  };

  return (
    <form className="stack" onSubmit={submit} noValidate>
      <PageHeader
        title="Nova execução"
        crumbs={[
          { label: "Datasets", to: "/" },
          { label: details.entry.name, to: `/datasets/${id}` },
          { label: "Pipeline robusto" },
        ]}
        lead="Configure o pipeline em quatro passos. Os valores iniciais reproduzem o padrão do notebook."
      />

      <Step n={1} title="Variáveis" hint="Escolha as séries que entram na análise.">
        <div className="row">
          <button type="button" onClick={() => setColumns(details.available_columns)}>
            Todas
          </button>
          <button type="button" onClick={() => setColumns(details.selected_columns)}>
            Padrão do dataset
          </button>
          <button type="button" onClick={() => setColumns([])}>
            Nenhuma
          </button>
          <span className="muted small">
            {columns.length} de {details.available_columns.length} selecionadas
          </span>
        </div>
        <div className="checks">
          {details.available_columns.map((c) => (
            <label key={c} className="chip-toggle">
              <input type="checkbox" checked={columns.includes(c)} onChange={() => setColumns(toggle(columns, c))} />
              {c}
            </label>
          ))}
        </div>
      </Step>

      <Step n={2} title="Objetivo da análise" hint="Define quais relações origem → destino serão avaliadas.">
        <RelationSelector
          nodes={columns}
          objective={objective}
          onObjectiveChange={setObjective}
          specific={specific}
          onSpecificChange={setSpecific}
          relationCount={relations.length}
        />
      </Step>

      <Step n={3} title="Métodos e esforço" hint="Mais métodos e bootstraps dão resultados mais estáveis, porém mais lentos.">
        <div className="row">
          <span className="field-label">Esforço</span>
          <div className="segmented" role="group" aria-label="Predefinições de esforço">
            {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
              <button key={key} type="button" aria-pressed={preset === key} onClick={() => applyPreset(key)} title={PRESETS[key].hint}>
                {PRESETS[key].label}
              </button>
            ))}
          </div>
          <span className="muted small">{preset === "custom" ? "personalizado" : PRESETS[preset].hint}</span>
        </div>
        <div className="row">
          <button type="button" onClick={() => setSelectedMethods(allMethodNames)}>
            Todos os métodos
          </button>
          <button type="button" onClick={() => setSelectedMethods([])}>
            Nenhum
          </button>
        </div>
        <div className="checks">
          {methods.data.map((m) => (
            <label key={m.name} className="chip-toggle">
              <input
                type="checkbox"
                checked={selectedMethods.includes(m.name)}
                onChange={() => setSelectedMethods(toggle(selectedMethods, m.name))}
              />
              {m.name}
            </label>
          ))}
        </div>
        <div className="form-grid">
          <Field label="Lag máximo" hint="Atraso máximo (em passos) considerado nas relações.">
            {(props) => (
              <input {...props} type="number" min={1} max={20} value={maxLag} onChange={(e) => setMaxLag(Number(e.target.value))} />
            )}
          </Field>
          <Field label={`Limiar do ensemble: ${threshold.toFixed(2)}`} hint="Probabilidade mínima para selecionar uma aresta.">
            {(props) => (
              <input {...props} type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            )}
          </Field>
        </div>
        <details className="advanced">
          <summary>Opções avançadas</summary>
          <div className="stack">
            <div className="form-grid">
              <Field label="Bootstraps" hint="Vazio = padrão do modo (4 rápido, 8 completo).">
                {(props) => (
                  <input
                    {...props}
                    type="number"
                    min={1}
                    max={100}
                    value={nBootstrap}
                    onChange={(e) => {
                      setNBootstrap(e.target.value);
                      setPreset("custom");
                    }}
                  />
                )}
              </Field>
              <Field label="Threads por execução" hint="Métodos rodam em paralelo; vazio = automático (até 4).">
                {(props) => (
                  <input {...props} type="number" min={1} max={8} value={parallelJobs} onChange={(e) => setParallelJobs(e.target.value)} />
                )}
              </Field>
              <Field label="Período sazonal" hint="Remove sazonalidade antes da análise; vazio = nenhum.">
                {(props) => <input {...props} type="number" min={2} value={decomposition} onChange={(e) => setDecomposition(e.target.value)} />}
              </Field>
              {details.trajectory_count > 1 && (
                <Field label={`Trajetória (0–${details.trajectory_count - 1})`} hint="Vazio = a padrão do catálogo.">
                  {(props) => (
                    <input
                      {...props}
                      type="number"
                      min={0}
                      max={details.trajectory_count - 1}
                      value={trajectory}
                      onChange={(e) => setTrajectory(e.target.value)}
                    />
                  )}
                </Field>
              )}
            </div>
            <div className="checks">
              <label className="check">
                <input
                  type="checkbox"
                  checked={quickMode}
                  onChange={(e) => {
                    setQuickMode(e.target.checked);
                    setPreset("custom");
                  }}
                />
                Modo rápido (busca até 3 métodos)
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
                  Evidência de painel (PCMCI em todas as trajetórias; roda em paralelo com a seleção)
                </label>
              )}
            </div>
          </div>
        </details>
      </Step>

      <Step
        n={4}
        title="Conhecimento especialista (opcional)"
        hint="Regras ajustam probabilidades; não alteram os dados. Use apenas conhecimento de domínio, nunca o gabarito."
      >
        <ExpertRulesEditor columns={columns} rules={rules} onChange={setRules} />
      </Step>

      {createRun.isError && <ErrorBox error={createRun.error} />}
      <div className="action-bar">
        <button type="submit" className="primary" disabled={problems.length > 0 || createRun.isPending}>
          {createRun.isPending ? "Enviando…" : "Executar pipeline"}
        </button>
        <span className="small">
          <strong>{columns.length}</strong> variáveis · <strong>{relations.length}</strong> relações ·{" "}
          <strong>{selectedMethods.length}</strong> métodos · {estimateMinutes("pipeline", { quickMode, methods: selectedMethods.length })}
        </span>
        {problems.length > 0 && (
          <span className="error small" role="alert">
            {problems[0]}
          </span>
        )}
      </div>
    </form>
  );
}
