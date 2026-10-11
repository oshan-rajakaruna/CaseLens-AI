import WorkspaceState from "../components/WorkspaceState.jsx";

export default function ReportsPage({ reports, state = "unavailable", errorMessage, onSelectReport }) {
  const records = Array.isArray(reports) ? reports : [];
  return (
    <div className="workspace-page" data-page="reports">
      <header className="workspace-page__header">
        <p className="eyebrow">Research library</p>
        <h1>Research reports</h1>
        <p>Open saved research reports when authenticated report APIs are connected.</p>
      </header>

      {state === "unavailable" && (
        <WorkspaceState
          state="unavailable"
          message={
            "Report listing requires secure report APIs and persistence. "
            + "No report list has been fetched."
          }
        />
      )}
      {state === "loading" && (
        <WorkspaceState state="loading" message="Loading reports from the configured service." />
      )}
      {state === "error" && <WorkspaceState state="error" message={errorMessage} />}
      {state === "ready" && records.length === 0 && (
        <WorkspaceState
          state="empty"
          message="The completed report request returned no report records."
        />
      )}
      {state === "ready" && records.length > 0 && (
        <ul className="record-list" aria-label="Research reports">
          {records.map((reportRecord) => (
            <li className="record-list__item" key={reportRecord.id}>
              <div>
                <p className="record-id">{reportRecord.id}</p>
                <h2>{reportRecord.title}</h2>
                {reportRecord.case_id && <p>Case ID: {reportRecord.case_id}</p>}
                {reportRecord.description && <p>{reportRecord.description}</p>}
              </div>
              {onSelectReport && (
                <button type="button" onClick={() => onSelectReport(reportRecord)}>
                  Open report
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
