import { NavLink, Route, Routes } from "react-router-dom";

import { DatasetPage } from "./features/datasets/DatasetPage";
import { DatasetsPage } from "./features/datasets/DatasetsPage";
import { NewRunPage } from "./features/runs/NewRunPage";
import { RunPage } from "./features/runs/RunPage";
import { RunsPage } from "./features/runs/RunsPage";

export function App() {
  return (
    <>
      <header className="topbar">
        <strong>Causal Discovery TS</strong>
        <nav>
          <NavLink to="/" end>
            Datasets
          </NavLink>
          <NavLink to="/runs">Execuções</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/datasets/:id" element={<DatasetPage />} />
          <Route path="/datasets/:id/new-run" element={<NewRunPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
          <Route path="*" element={<p>Página não encontrada.</p>} />
        </Routes>
      </main>
    </>
  );
}
