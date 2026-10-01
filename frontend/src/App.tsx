import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";

import { isTerminal, useRuns } from "./api/hooks";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { EmptyState } from "./components/ui";
import { ExperienceToggle } from "./lib/experience";
import { AtlasChatPage, AtlasExperimentPage } from "./features/atlas/AtlasExperimentPage";
import { ChatDock } from "./features/assistant/ChatDock";
import { AtlasPage } from "./features/atlas/AtlasPage";
import { BenchmarkPage } from "./features/benchmark/BenchmarkPage";
import { DatasetPage } from "./features/datasets/DatasetPage";
import { DatasetsPage } from "./features/datasets/DatasetsPage";
import { ReplicatedValidationPage } from "./features/datasets/ReplicatedValidationPage";
import { NewRunPage } from "./features/runs/NewRunPage";
import { RunPage } from "./features/runs/RunPage";
import { RunsPage } from "./features/runs/RunsPage";

function Brand() {
  return (
    <Link to="/" className="brand" aria-label="Causal Discovery TS — início">
      <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden>
        <circle cx="5" cy="6" r="3" stroke="currentColor" strokeWidth="2" />
        <circle cx="21" cy="6" r="3" stroke="currentColor" strokeWidth="2" />
        <circle cx="13" cy="21" r="3" stroke="currentColor" strokeWidth="2" />
        <path d="M8 6h10M6.5 8.6l5 9M19.5 8.6l-5 9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      Causal Discovery TS
    </Link>
  );
}

function RunsNavLink() {
  const runs = useRuns();
  const active = runs.data?.filter((run) => !isTerminal(run.status)).length ?? 0;
  return (
    <NavLink to="/runs">
      Execuções
      {active > 0 && (
        <span className="pill" title={`${active} execução(ões) em andamento`} aria-label={`${active} em andamento`}>
          {active}
        </span>
      )}
    </NavLink>
  );
}

function NotFound() {
  return (
    <EmptyState title="Página não encontrada">
      <Link to="/">Voltar aos datasets</Link>
    </EmptyState>
  );
}

export function App() {
  const location = useLocation();
  return (
    <>
      <a className="skip-link" href="#conteudo">
        Ir para o conteúdo
      </a>
      <header className="topbar">
        <Brand />
        <nav aria-label="Principal">
          <NavLink to="/" end>
            Datasets
          </NavLink>
          <RunsNavLink />
          <NavLink to="/benchmark">Benchmark</NavLink>
          <NavLink to="/atlas">Atlas</NavLink>
        </nav>
        <div className="topbar-end">
          <ExperienceToggle />
        </div>
      </header>
      <main id="conteudo">
        <ErrorBoundary resetKey={location.pathname}>
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
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </main>
      <ChatDock />
    </>
  );
}
