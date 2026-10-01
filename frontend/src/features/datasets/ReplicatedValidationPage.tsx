import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { useCreateReplicatedValidation, useDataset } from "../../api/hooks";
import { Card, ErrorBox, PageHeader, PageSkeleton } from "../../components/ui";
import { AdvancedOnly, BeginnerHint } from "../../lib/experience";
import { MethodChecklist } from "../runs/MethodChecklist";

/** Validação estatística pareada (Wilcoxon + Holm + IC + taxa de vitórias) sobre réplicas independentes. */
export function ReplicatedValidationPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const dataset = useDataset(id);
  const create = useCreateReplicatedValidation();
  const [replicates, setReplicates] = useState(10);
  const [bootstraps, setBootstraps] = useState(10000);
  const [alpha, setAlpha] = useState(0.05);
  const [minGain, setMinGain] = useState(0.05);
  const [winRate, setWinRate] = useState(0.7);
  const [minConfirmatory, setMinConfirmatory] = useState(10);
  const [maxLag, setMaxLag] = useState(2);
  const [quickMode, setQuickMode] = useState(false);
  const [parallelReplicas, setParallelReplicas] = useState("");
  const [methods, setMethods] = useState<string[]>([]);

  if (dataset.isPending) return <PageSkeleton />;
  if (dataset.isError) return <ErrorBox error={dataset.error} />;
  const details = dataset.data;

  if (!details.supports_replicates || !details.has_ground_truth) {
    return (
      <div className="stack">
        <h1>Validação com réplicas</h1>
        <p className="error">
          Este dataset não expõe réplicas independentes com grafo verdadeiro (hoje só o CausalTime). Uma série contínua única
          não permite o protocolo pareado.
        </p>
      </div>
    );
  }

  const estimate = quickMode ? replicates : replicates * 5;
  const canSubmit = methods.length >= 2 && !create.isPending;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    create.mutate(
      {
        dataset_id: id,
        n_replicates: replicates,
        statistical_bootstraps: bootstraps,
        significance_level: alpha,
        minimum_precision_gain: minGain,
        minimum_win_rate: winRate,
        min_confirmatory_replicates: minConfirmatory,
        max_lag: maxLag,
        quick_mode: quickMode,
        parallel_replicas: parallelReplicas ? Number(parallelReplicas) : null,
        methods,
        ensemble_threshold: 0.5,
        replicate_seed: 2029,
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <PageHeader
        title="Validação com réplicas"
        crumbs={[{ label: "Datasets", to: "/" }, { label: details.entry.name, to: `/datasets/${id}` }, { label: "Validação com réplicas" }]}
        lead={
          <>
        Uma única execução não é evidência estatística de superioridade. Esta análise reexecuta o ENSEMBLE_AUTO em
        trajetórias independentes (as já usadas como desenvolvimento/holdout ficam fora da amostra) e compara a precisão
        pareada contra cada método e contra o ensemble completo.
          </>
        }
      />
      <BeginnerHint>
        Uma execução única pode acertar por sorte. Aqui a mesma análise é repetida em várias séries independentes e um
        teste estatístico diz se o ensemble é melhor de forma consistente. Comece com poucas réplicas no modo rápido.
      </BeginnerHint>
      <Card title="Réplicas e estatística">
        <div className="form-grid">
          <label>
            Réplicas
            <input type="number" min={2} max={100} value={replicates} onChange={(e) => setReplicates(Number(e.target.value))} />
          </label>
          <AdvancedOnly>
          <label>
            Bootstraps do IC
            <input type="number" min={100} max={100000} step={100} value={bootstraps} onChange={(e) => setBootstraps(Number(e.target.value))} />
          </label>
          <label>
            Nível de significância
            <input type="number" min={0.001} max={0.5} step={0.01} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
          </label>
          <label>
            Ganho mínimo de precisão
            <input type="number" min={0} max={1} step={0.01} value={minGain} onChange={(e) => setMinGain(Number(e.target.value))} />
          </label>
          <label>
            Taxa mínima de vitórias
            <input type="number" min={0} max={1} step={0.05} value={winRate} onChange={(e) => setWinRate(Number(e.target.value))} />
          </label>
          <label>
            Réplicas mínimas (confirmatório)
            <input type="number" min={2} value={minConfirmatory} onChange={(e) => setMinConfirmatory(Number(e.target.value))} />
          </label>
          <label>
            Lag máximo
            <input type="number" min={1} max={20} value={maxLag} onChange={(e) => setMaxLag(Number(e.target.value))} />
          </label>
          <label>
            Réplicas em paralelo (vazio = automático)
            <input type="number" min={1} max={8} value={parallelReplicas} onChange={(e) => setParallelReplicas(e.target.value)} />
            <span className="muted small">O orçamento de CPU é dividido entre elas; os resultados são idênticos aos da execução sequencial.</span>
          </label>
          </AdvancedOnly>
        </div>
        <label className="check">
          <input type="checkbox" checked={quickMode} onChange={(e) => setQuickMode(e.target.checked)} />
          Modo rápido (busca até 3 métodos, menos bootstraps por réplica)
        </label>
        <p className="muted">
          Custo estimado: ~{estimate} min ({quickMode ? "modo rápido" : "≈5 min por réplica com a busca completa"}). A
          execução pode ser cancelada entre réplicas.
        </p>
      </Card>
      <Card title="Métodos candidatos">
        <MethodChecklist value={methods} onChange={setMethods} />
      </Card>
      {create.isError && <ErrorBox error={create.error} />}
      <div>
        <button type="submit" className="primary" disabled={!canSubmit}>
          {create.isPending ? "Enviando…" : "Executar validação"}
        </button>
      </div>
    </form>
  );
}
