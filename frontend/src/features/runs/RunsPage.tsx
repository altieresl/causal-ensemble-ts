import { Link } from "react-router-dom";

import { useRuns } from "../../api/hooks";
import { Card, ErrorBox, Spinner, StatusBadge, TableWrap } from "../../components/ui";
import { formatDateTime, formatDuration } from "../../lib/format";

export function RunsPage() {
  const runs = useRuns();
  return (
    <div className="stack">
      <h1>Execuções</h1>
      {runs.isPending && <Spinner />}
      {runs.isError && <ErrorBox error={runs.error} />}
      {runs.data && (
        <Card>
          <TableWrap>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Dataset</th>
                  <th>Status</th>
                  <th>Criada</th>
                  <th>Duração</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.length === 0 && (
                  <tr>
                    <td colSpan={5} className="muted">
                      Nenhuma execução ainda. Escolha um <Link to="/">dataset</Link>.
                    </td>
                  </tr>
                )}
                {runs.data.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link to={`/runs/${run.id}`}>{run.id}</Link>
                    </td>
                    <td>{String(run.params.dataset_id)}</td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td>{formatDateTime(run.created_at)}</td>
                    <td>{formatDuration(run.started_at, run.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableWrap>
        </Card>
      )}
    </div>
  );
}
