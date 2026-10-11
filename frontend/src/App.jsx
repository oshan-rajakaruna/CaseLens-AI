import { lazy, Suspense } from "react";

import AppHeader from "./components/AppHeader.jsx";
import CaseDetailPage from "./pages/CaseDetailPage.jsx";
import CasesPage from "./pages/CasesPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import HomePage from "./pages/HomePage.jsx";
import ReportPage from "./pages/ReportPage.jsx";
import ReportsPage from "./pages/ReportsPage.jsx";

const ReportPreviewPage = import.meta.env.DEV
  ? lazy(() => import("./pages/ReportPreviewPage.jsx"))
  : null;

export function shouldShowLegalReportPreview({ isDevelopment, search }) {
  return isDevelopment && new URLSearchParams(search).get("preview") === "legal-report";
}

export function pageFromSearch(search) {
  const requested = new URLSearchParams(search).get("page");
  return ["dashboard", "cases", "case", "reports", "report"].includes(requested)
    ? requested
    : "home";
}

function WorkspacePage({ page }) {
  if (page === "dashboard") return <DashboardPage />;
  if (page === "cases") return <CasesPage />;
  if (page === "case") return <CaseDetailPage />;
  if (page === "reports") return <ReportsPage />;
  if (page === "report") return <ReportPage />;
  return <HomePage />;
}

export default function App() {
  const showPreview = shouldShowLegalReportPreview({
    isDevelopment: import.meta.env.DEV,
    search: window.location.search,
  });
  const activePage = pageFromSearch(window.location.search);

  return (
    <div className="app-shell">
      <AppHeader activePage={showPreview ? "preview" : activePage} />
      <main>
        {showPreview && ReportPreviewPage ? (
          <Suspense fallback={<p className="preview-loading">Loading development preview…</p>}>
            <ReportPreviewPage />
          </Suspense>
        ) : (
          <WorkspacePage page={activePage} />
        )}
      </main>
    </div>
  );
}
