import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useDatasets, useDeleteDataset, useUploadDataset } from "../../api/hooks";
import { useToast } from "../../components/toast";
import { Badge, Card, ConfirmButton, EmptyState, ErrorBox, PageHeader, PageSkeleton } from "../../components/ui";

export function DatasetsPage() {
  const datasets = useDatasets();
  const upload = useUploadDataset();
  const remove = useDeleteDataset();
  const { notify } = useToast();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dateColumn, setDateColumn] = useState("");
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (datasets.data ?? []).filter(
      (d) => !needle || `${d.name} ${d.description} ${d.id}`.toLowerCase().includes(needle),
    );
  }, [datasets.data, query]);

  const send = () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    upload.mutate(
      { file, dateColumn: dateColumn.trim() || undefined },
      {
        onSuccess: (created) => {
          notify(`Dataset “${created.name}” enviado.`, "success");
          if (fileInput.current) fileInput.current.value = "";
          setFileName(null);
          setDateColumn("");
        },
      },
    );
  };

  const pick = (files: FileList | null) => {
    if (!files?.length || !fileInput.current) return;
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    fileInput.current.files = transfer.files;
    setFileName(files[0].name);
  };

  return (
    <div className="stack">
      <PageHeader
        title="Datasets"
        lead="Escolha uma série temporal para perfilar e analisar, ou envie o seu CSV. O perfil indica quais algoritmos de descoberta causal são compatíveis com os dados."
      />

      <Card title="Enviar CSV">
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            send();
          }}
        >
          <div
            className={`dropzone${dragging ? " over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pick(e.dataTransfer.files);
            }}
          >
            {fileName ? (
              <strong>{fileName}</strong>
            ) : (
              <>Arraste um arquivo .csv aqui ou use o seletor abaixo.</>
            )}
          </div>
          <div className="row">
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              required
              aria-label="Arquivo CSV"
              onChange={(e) => setFileName(e.target.files?.[0]?.name ?? null)}
            />
            <label className="check">
              Coluna temporal (opcional)
              <input value={dateColumn} onChange={(e) => setDateColumn(e.target.value)} placeholder="date" />
            </label>
            <button type="submit" className="primary" disabled={upload.isPending || !fileName}>
              {upload.isPending ? "Enviando…" : "Enviar"}
            </button>
          </div>
        </form>
        {upload.isError && <ErrorBox error={upload.error} />}
        <p className="muted small">Requer ao menos 2 colunas numéricas e 30 linhas (limite de tamanho definido no servidor).</p>
      </Card>

      <div className="row between">
        <h2>Disponíveis{datasets.data ? ` (${visible.length}${query ? ` de ${datasets.data.length}` : ""})` : ""}</h2>
        <input
          type="search"
          placeholder="Buscar dataset…"
          aria-label="Buscar dataset"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {datasets.isPending && <PageSkeleton />}
      {datasets.isError && <ErrorBox error={datasets.error} />}
      {datasets.data && visible.length === 0 && (
        <EmptyState title="Nenhum dataset encontrado">Tente outro termo de busca.</EmptyState>
      )}
      <div className="grid">
        {visible.map((dataset) => (
          <Card key={dataset.id} className="interactive">
            <div className="row between">
              <h2>{dataset.name}</h2>
              <Badge tone={dataset.origin === "upload" ? "warn" : "muted"}>
                {dataset.origin === "upload" ? "enviado" : "embutido"}
              </Badge>
            </div>
            <p className="muted">{dataset.description}</p>
            <div className="row">
              <Link to={`/datasets/${dataset.id}`} className="button primary">
                Abrir
              </Link>
              {dataset.origin === "upload" && (
                <ConfirmButton
                  onConfirm={() => remove.mutate(dataset.id, { onSuccess: () => notify("Dataset removido.") })}
                  confirmLabel="Confirmar remoção"
                  disabled={remove.isPending}
                >
                  Remover
                </ConfirmButton>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
