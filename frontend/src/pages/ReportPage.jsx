import LegalReportView from "../components/report/LegalReportView.jsx";
import WorkspaceState from "../components/WorkspaceState.jsx";

export default function ReportPage({ report }) {
  if (!report) {
    return (
      <div className="workspace-page" data-page="report-detail">
        <header className="workspace-page__header">
          <p className="eyebrow">Research library</p>
          <h1>Report details</h1>
        </header>
        <WorkspaceState
          state="unavailable"
          message={
            "No validated report was selected. Report content cannot be inferred "
            + "or generated in the browser."
          }
        />
      </div>
    );
  }
  return <LegalReportView report={report} />;
}
