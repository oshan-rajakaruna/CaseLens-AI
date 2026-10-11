import { Detail, DetailList } from "./ReportPrimitives.jsx";

export default function CaseInformationSection({ information }) {
  const caseRecord = information?.case;
  return (
    <section className="report-section" aria-labelledby="case-information-title">
      <div className="section-heading">
        <p className="section-kicker">Research subject</p>
        <h2 id="case-information-title">Case information</h2>
      </div>
      {!caseRecord ? (
        <p className="empty-state">Case information is unavailable.</p>
      ) : (
        <DetailList>
          <Detail label="Case title" value={caseRecord.title} />
          <Detail label="Case ID" value={caseRecord.id} />
          <Detail label="Case description" value={caseRecord.description} />
          <Detail label="Research query" value={information.query} />
          <Detail label="Legal issue" value={information.legal_issue} />
        </DetailList>
      )}
    </section>
  );
}
