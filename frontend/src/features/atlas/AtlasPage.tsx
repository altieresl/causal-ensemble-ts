import { useState } from "react";

import { useAlgorithms, useAsk } from "../../api/hooks";
import type { Algorithm } from "../../api/types";
import { Badge, Card, ErrorBox, Spinner, TableWrap } from "../../components/ui";

const yesNo = (value: boolean) => (value ? "sim" : "não");

function AlgorithmRow({ algorithm }: { algorithm: Algorithm }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr>
        <td>
          <button className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
            {algorithm.name}
          </button>
        </td>
        <td>{algorithm.family}</td>
        <td>{algorithm.temporal_handling}</td>
        <td>{yesNo(algorithm.handles_nonlinearity)}</td>
        <td>{yesNo(algorithm.handles_latent_confounders)}</td>
        <td>
          {algorithm.implemented_in_framework ? (
            <Badge tone="ok">{algorithm.framework_method_name}</Badge>
          ) : (
            <Badge tone="muted">não implementado</Badge>
          )}
        </td>
        <td>
          <Badge tone={algorithm.verification === "verified" ? "ok" : "warn"}>{algorithm.verification}</Badge>
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={7}>
            <p>
              <strong>Saída:</strong> {algorithm.output_type} · <strong>Efeitos instantâneos:</strong>{" "}
              {yesNo(algorithm.handles_contemporaneous_effects)}
            </p>
            <p><strong>Premissas:</strong></p>
            <ul>
              {algorithm.assumptions.map((a) => (
                <li key={a.id}>
                  <code>{a.id}</code> {a.required ? "(obrigatória)" : "(opcional)"} — {a.statement}
                </li>
              ))}
            </ul>
            <p><strong>Referências:</strong></p>
            <ul>
              {algorithm.references.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </td>
        </tr>
      )}
    </>
  );
}

function AskCard() {
  const ask = useAsk();
  const [query, setQuery] = useState("");
  const [generate, setGenerate] = useState(false);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    ask.mutate({ query: query.trim(), k: 4, generate });
  };

  return (
    <Card title="Perguntar à base de conhecimento (RAG local)">
      <form className="stack" onSubmit={submit}>
        <div className="row">
          <input
            className="grow"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ex.: quais métodos toleram confundidores latentes?"
            aria-label="Pergunta"
            minLength={3}
            maxLength={500}
            required
          />
          <button type="submit" disabled={ask.isPending || query.trim().length < 3}>
            {ask.isPending ? "Buscando…" : "Buscar"}
          </button>
        </div>
        <label className="check">
          <input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} />
          Gerar resposta com o Ollama local (senão, apenas os trechos recuperados)
        </label>
      </form>
      {ask.isError && <ErrorBox error={ask.error} />}
      {ask.data && (
        <div className="stack">
          {ask.data.answer && (
            <blockquote>
              <strong>Resposta:</strong> {ask.data.answer}
            </blockquote>
          )}
          {ask.data.error && <p className="error">{ask.data.error}</p>}
          <p className="muted">Trechos recuperados (similaridade TF-IDF):</p>
          <ul className="plain">
            {ask.data.retrieved.map((chunk, index) => (
              <li key={index}>
                <strong>
                  {chunk.algorithm_id} / {chunk.section}
                </strong>{" "}
                <span className="muted">({chunk.score.toFixed(3)})</span>
                <p>{chunk.text}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

export function AtlasPage() {
  const algorithms = useAlgorithms();
  return (
    <div className="stack">
      <h1>Atlas de algoritmos</h1>
      <p className="muted">
        Fichas com premissas declaradas e referências. A recomendação de métodos usa só essas premissas e o perfil dos
        dados — nunca o gabarito.
      </p>
      {algorithms.isPending && <Spinner />}
      {algorithms.isError && <ErrorBox error={algorithms.error} />}
      {algorithms.data && (
        <Card>
          <TableWrap>
            <table>
              <thead>
                <tr>
                  <th>Algoritmo</th>
                  <th>Família</th>
                  <th>Tratamento temporal</th>
                  <th>Não linear</th>
                  <th>Confundidores latentes</th>
                  <th>No framework</th>
                  <th>Verificação</th>
                </tr>
              </thead>
              <tbody>
                {algorithms.data.map((algorithm) => (
                  <AlgorithmRow key={algorithm.id} algorithm={algorithm} />
                ))}
              </tbody>
            </table>
          </TableWrap>
        </Card>
      )}
      <AskCard />
    </div>
  );
}
