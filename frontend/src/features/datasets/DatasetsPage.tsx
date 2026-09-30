import { useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useDatasets, useDeleteDataset, useUploadDataset } from "../../api/hooks";
import { Badge, Card, ErrorBox, Spinner } from "../../components/ui";

export function DatasetsPage() {
  const datasets = useDatasets();
  const upload = useUploadDataset();
  const remove = useDeleteDataset();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dateColumn, setDateColumn] = useState("");

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    upload.mutate(
      { file, dateColumn: dateColumn.trim() || undefined },
      {
        onSuccess: () => {
          if (fileInput.current) fileInput.current.value = "";
          setDateColumn("");
        },
      },
    );
  };

  return (
    <div className="stack">
      <h1>Datasets</h1>
      <Card title="Enviar CSV">
        <form onSubmit={onSubmit} className="row">
          <input ref={fileInput} type="file" accept=".csv,text/csv" required aria-label="Arquivo CSV" />
          <label>
            Coluna temporal (opcional){" "}
            <input value={dateColumn} onChange={(e) => setDateColumn(e.target.value)} placeholder="date" />
          </label>
          <button type="submit" disabled={upload.isPending}>
            {upload.isPending ? "Enviando…" : "Enviar"}
          </button>
        </form>
        {upload.isError && <ErrorBox error={upload.error} />}
        <p className="muted">Requer ao menos 2 colunas numéricas e 30 linhas.</p>
      </Card>

      {datasets.isPending && <Spinner />}
      {datasets.isError && <ErrorBox error={datasets.error} />}
      <div className="grid">
        {datasets.data?.map((dataset) => (
          <Card key={dataset.id} title={dataset.name}>
            <p className="muted">{dataset.description}</p>
            <div className="row">
              <Badge tone={dataset.origin === "upload" ? "warn" : "muted"}>
                {dataset.origin === "upload" ? "enviado" : "embutido"}
              </Badge>
              <Link to={`/datasets/${dataset.id}`} className="button">
                Abrir
              </Link>
              {dataset.origin === "upload" && (
                <button className="danger" onClick={() => remove.mutate(dataset.id)} disabled={remove.isPending}>
                  Remover
                </button>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
