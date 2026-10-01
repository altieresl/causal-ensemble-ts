import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import { App } from "./App";
import { ToastProvider } from "./components/toast";
import { AssistantProvider } from "./features/assistant/AssistantProvider";
import { ExperienceProvider } from "./lib/experience";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ExperienceProvider>
        <ToastProvider>
          <BrowserRouter>
            <AssistantProvider>
              <App />
            </AssistantProvider>
          </BrowserRouter>
        </ToastProvider>
      </ExperienceProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
