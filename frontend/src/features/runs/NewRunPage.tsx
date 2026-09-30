import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { useCreateRun, useDataset, useMethods } from "../../api/hooks";
import type { ExpertRule } from "../../api/types";
import { Card, ErrorBox, Spinner } from "../../components/ui";
import { ExpertRulesEditor } from "./ExpertRulesEditor";

const toggle = (list: string[], item: string) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);

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
  const [rules, setRules] = useState<ExpertRule[]>([]);

  useEffect(() => {
    if (dataset.data) {
      setColumns(dataset.data.selected_columns);
      setMaxLag(dataset.data.default_max_lag);
    }
  }, [dataset.data]);
  useEffect(() => {
    if (methods.data) setSelectedMethods(methods.data.map((m) => m.name));
  }, [methods.data]);

  if (dataset.isPending || methods.isPending) return <Spinner />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  if (methods.isError) return <ErrorBox error={methods.error} />;

  const validRules = rules.filter((r) => r.source && r.target && r.source !== r.target);
  const canSubmit = columns.length >= 2 && selectedMethods.length >= 2 && !createRun.isPending;

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
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <h1>Nova execução — {dataset.data.entry.name}</h1>

      <Card title="Variáveis">
        <div className="checks">
          {dataset.data.available_columns.map((c) => (
            <label key={c} className="check">
              <input type="checkbox" checked={columns.includes(c)} onChange={() => setColumns(toggle(columns, c))} />
              {c}
            </label>
          ))}
        </div>
        {columns.length < 2 && <p className="error">Selecione ao menos 2 variáveis.</p>}
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
