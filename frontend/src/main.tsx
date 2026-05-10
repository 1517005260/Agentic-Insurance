import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.tsx";
import { ErrorBoundary, installGlobalErrorReporters } from "./components/ErrorBoundary";
import "./index.css";

installGlobalErrorReporters();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
