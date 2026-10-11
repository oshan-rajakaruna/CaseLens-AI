import WorkspaceState from "../components/WorkspaceState.jsx";

export default function CaseDetailPage({ caseRecord }) {
  return (
    <div className="workspace-page" data-page="case-detail">
      <header className="workspace-page__header">
        <p className="eyebrow">Case workspace</p>
        <h1>Case details</h1>
      </header>
      {!caseRecord ? (
        <WorkspaceState
          state="unavailable"
          message="No validated case record was selected. Case details cannot be inferred."
        />
      ) : (
        <article className="workspace-section case-detail">
          <p className="record-id">{caseRecord.id}</p>
          <h2>{caseRecord.title}</h2>
          {caseRecord.description ? (
            <p>{caseRecord.description}</p>
          ) : (
            <p className="muted-text">No case description was supplied.</p>
          )}
          <div className="action-unavailable" role="note">
            Document management and research execution will be enabled only after secure
            backend contracts are available.
          </div>
        </article>
      )}
    </div>
  );
}
