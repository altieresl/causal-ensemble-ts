import { NavLink, Route, Routes } from "react-router-dom";

import { AtlasPage } from "./features/atlas/AtlasPage";
import { AtlasChatPage, AtlasExperimentPage } from "./features/atlas/AtlasExperimentPage";
import { BenchmarkPage } from "./features/benchmark/BenchmarkPage";
import { ReplicatedValidationPage } from "./features/datasets/ReplicatedValidationPage";
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
          <NavLink to="/benchmark">Benchmark</NavLink>
          <NavLink to="/atlas">Atlas</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/datasets/:id" element={<DatasetPage />} />
          <Route path="/datasets/:id/new-run" element={<NewRunPage />} />
          <Route path="/datasets/:id/validation" element={<ReplicatedValidationPage />} />
          <Route path="/datasets/:id/atlas-experiment" element={<AtlasExperimentPage />} />
          <Route path="/datasets/:id/atlas-chat" element={<AtlasChatPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
          <Route path="/atlas" element={<AtlasPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
          <Route path="*" element={<p>Página não encontrada.</p>} />
        </Routes>
      </main>
    </>
  );
}
