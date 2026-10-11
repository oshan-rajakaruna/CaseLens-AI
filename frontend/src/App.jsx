import { lazy, Suspense } from "react";

import AppHeader from "./components/AppHeader.jsx";
import HomePage from "./pages/HomePage.jsx";

const ReportPreviewPage = import.meta.env.DEV
  ? lazy(() => import("./pages/ReportPreviewPage.jsx"))
  : null;

export function shouldShowLegalReportPreview({ isDevelopment, search }) {
  return isDevelopment && new URLSearchParams(search).get("preview") === "legal-report";
}

export default function App() {
  const showPreview = shouldShowLegalReportPreview({
    isDevelopment: import.meta.env.DEV,
    search: window.location.search,
  });

  return (
    <div className="app-shell">
      <AppHeader />
      <main>
        {showPreview && ReportPreviewPage ? (
          <Suspense fallback={<p className="preview-loading">Loading development preview…</p>}>
            <ReportPreviewPage />
          </Suspense>
        ) : (
          <HomePage />
        )}
      </main>
    </div>
  );
}
