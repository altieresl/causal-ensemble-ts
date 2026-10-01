import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useCreateBenchmark } from "../../api/hooks";
import { Card, ErrorBox, PageHeader } from "../../components/ui";
import { AdvancedOnly, BeginnerHint } from "../../lib/experience";
import { MethodChecklist } from "../runs/MethodChecklist";

/** Benchmark sintético (gerador com estrutura conhecida) + robustez a mudança no regime de ruído. */
export function BenchmarkPage() {
  const navigate = useNavigate();
  const create = useCreateBenchmark();
  const [nSamples, setNSamples] = useState(500);
  const [noise, setNoise] = useState(3);
  const [indexChange, setIndexChange] = useState(250);
  const [nBootstrap, setNBootstrap] = useState(20);
  const [maxLag, setMaxLag] = useState(2);
  const [methods, setMethods] = useState<string[]>([]);

  const invalidChange = indexChange < 0 || indexChange >= nSamples;
  const canSubmit = methods.length >= 2 && !invalidChange && !create.isPending;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    create.mutate(
      {
        n_samples: nSamples,
        noise_multiplier: noise,
        index_change: indexChange,
        n_bootstrap: nBootstrap,
        max_lag: maxLag,
        methods,
      },
      { onSuccess: (run) => navigate(`/runs/${run.id}`) },
    );
  };

  return (
    <form className="stack" onSubmit={submit}>
      <PageHeader
        title="Benchmark sintético"
        crumbs={[{ label: "Benchmark" }]}
        lead={
          <>
        Gera séries com estrutura causal conhecida, roda a seleção robusta (consenso de 2 entre 3 métodos) e repete
        após multiplicar o ruído a partir de um ponto. O gabarito é usado só na avaliação, depois da seleção.
          </>
        }
      />
      <Card title="Série">
        <div className="form-grid">
          <label>
            Amostras
            <input type="number" min={100} max={5000} value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
          </label>
          <label>
            Multiplicador de ruído
            <input type="number" min={0} step={0.5} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
          </label>
          <label>
            Índice da mudança de regime
            <input type="number" min={0} value={indexChange} onChange={(e) => setIndexChange(Number(e.target.value))} />
          </label>
        </div>
        {invalidChange && <p className="error">O índice deve estar dentro da série (0 a {nSamples - 1}).</p>}
      </Card>
      <BeginnerHint>
        Este teste responde: “se o gerador dos dados é conhecido, a análise recupera as relações certas? E continua
        acertando quando os dados ficam muito mais ruidosos?”. Os valores padrão já servem.
      </BeginnerHint>
      <Card title="Seleção robusta">
        <AdvancedOnly>
        <div className="form-grid">
          <label>
            Bootstraps
            <input type="number" min={1} max={100} value={nBootstrap} onChange={(e) => setNBootstrap(Number(e.target.value))} />
          </label>
          <label>
            Lag máximo
            <input type="number" min={1} max={10} value={maxLag} onChange={(e) => setMaxLag(Number(e.target.value))} />
          </label>
        </div>
        </AdvancedOnly>
        <MethodChecklist value={methods} onChange={setMethods} />
        <p className="muted">Com todos os métodos o custo é alto (56 combinações de 3 métodos, duas séries).</p>
      </Card>
      {create.isError && <ErrorBox error={create.error} />}
      <div>
        <button type="submit" className="primary" disabled={!canSubmit}>
          {create.isPending ? "Enviando…" : "Executar benchmark"}
        </button>
      </div>
    </form>
  );
}
