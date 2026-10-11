import WorkspaceState from "../components/WorkspaceState.jsx";

export default function CasesPage({ cases, state = "unavailable", onSelectCase }) {
  if (state !== "ready") {
    return (
      <WorkspacePageHeader>
        <WorkspaceState
          state={state}
          message={
            state === "unavailable"
              ? "Case management requires authenticated case APIs and database "
                + "persistence. No case list has been fetched."
              : undefined
          }
        />
      </WorkspacePageHeader>
    );
  }

  const records = Array.isArray(cases) ? cases : [];
  return (
    <WorkspacePageHeader>
      {records.length === 0 ? (
        <WorkspaceState
          state="empty"
          message="The completed case request returned no case records."
        />
      ) : (
        <ul className="record-list" aria-label="Cases">
          {records.map((caseRecord) => (
            <li className="record-list__item" key={caseRecord.id}>
              <div>
                <p className="record-id">{caseRecord.id}</p>
                <h2>{caseRecord.title}</h2>
                {caseRecord.description && <p>{caseRecord.description}</p>}
              </div>
              {onSelectCase && (
                <button type="button" onClick={() => onSelectCase(caseRecord)}>
                  View case
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </WorkspacePageHeader>
  );
}

function WorkspacePageHeader({ children }) {
  return (
    <div className="workspace-page" data-page="cases">
      <header className="workspace-page__header">
        <p className="eyebrow">Case workspace</p>
        <h1>Cases</h1>
        <p>Review validated case records when the secure case-management service is connected.</p>
      </header>
      {children}
    </div>
  );
}
