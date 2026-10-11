import {
  confidenceText,
  Detail,
  DetailList,
  EvidenceReference,
  TextList,
} from "./ReportPrimitives.jsx";
import StatusBadge from "./StatusBadge.jsx";

function FindingCard({ title, children }) {
  return (
    <article className="finding-card">
      <h4>{title}</h4>
      {children}
    </article>
  );
}

function AnalysisDocument({ result, index }) {
  const hasFindings = [
    result.facts,
    result.clauses,
    result.legal_issues,
    result.timeline,
    result.contradiction_candidates,
  ].some((items) => Array.isArray(items) && items.length > 0);

  return (
    <article className="analysis-document">
      <div className="analysis-document__heading">
        <div>
          <p className="section-kicker">Analysis document {index + 1}</p>
          <h3>{result.document_id}</h3>
        </div>
        <StatusBadge label="Analysis status" value={result.status} />
      </div>

      {!hasFindings && (
        <p className="empty-state">No analysis findings were recorded for this document.</p>
      )}

      {result.facts?.map((fact, factIndex) => (
        <FindingCard title={`Extracted fact ${factIndex + 1}`} key={`fact-${factIndex}`}>
          <p className="finding-text">{fact.statement}</p>
          <div className="status-row">
            <StatusBadge label="Epistemic status" value={fact.epistemic_status} />
            <StatusBadge label="Verification" value={fact.verification_status} />
          </div>
          <DetailList>
            <Detail label="Category" value={fact.category} />
            <Detail label="Extraction confidence" value={confidenceText(fact.confidence)} />
            <Detail label="Method" value={fact.method} />
          </DetailList>
          <EvidenceReference source={fact.source} />
        </FindingCard>
      ))}

      {result.clauses?.map((clause, clauseIndex) => (
        <FindingCard title={`Extracted clause ${clauseIndex + 1}`} key={`clause-${clauseIndex}`}>
          <p className="finding-text">{clause.text}</p>
          <StatusBadge label="Epistemic status" value={clause.epistemic_status} />
          <DetailList>
            <Detail label="Clause type" value={clause.clause_type} />
            <Detail label="Extraction confidence" value={confidenceText(clause.confidence)} />
            <Detail label="Deadline" value={clause.deadline_text} />
          </DetailList>
          <EvidenceReference source={clause.source} />
        </FindingCard>
      ))}

      {result.legal_issues?.map((issue, issueIndex) => (
        <FindingCard title={`Legal issue candidate ${issueIndex + 1}`} key={`issue-${issueIndex}`}>
          <p className="finding-text">{issue.label}</p>
          {issue.description && <p>{issue.description}</p>}
          <div className="status-row">
            <StatusBadge label="Review status" value={issue.review_status} />
            <StatusBadge label="Epistemic status" value={issue.epistemic_status} />
          </div>
          <DetailList>
            <Detail label="Confidence" value={confidenceText(issue.confidence)} />
            <Detail label="Method" value={issue.method} />
          </DetailList>
          {issue.sources?.map((source, sourceIndex) => (
            <EvidenceReference
              key={`issue-source-${sourceIndex}`}
              label={`Evidence source ${sourceIndex + 1}`}
              source={source}
            />
          ))}
        </FindingCard>
      ))}

      {result.timeline?.map((event, eventIndex) => (
        <FindingCard title={`Timeline event ${eventIndex + 1}`} key={`event-${eventIndex}`}>
          <p className="finding-text">{event.description}</p>
          <div className="status-row">
            <StatusBadge label="Epistemic status" value={event.epistemic_status} />
            <StatusBadge label="Verification" value={event.verification_status} />
          </div>
          <DetailList>
            <Detail label="Date as supplied" value={event.date_text} />
            <Detail label="Normalized date" value={event.normalized_date} />
            <Detail label="Confidence" value={confidenceText(event.confidence)} />
            <Detail label="Review warning" value={event.review_warning} />
          </DetailList>
          <EvidenceReference source={event.source} />
        </FindingCard>
      ))}

      {result.contradiction_candidates?.map((candidate, candidateIndex) => (
        <FindingCard
          title={`Contradiction candidate ${candidateIndex + 1}`}
          key={`contradiction-${candidateIndex}`}
        >
          <p className="finding-text">{candidate.description}</p>
          <StatusBadge label="Review status" value={candidate.review_status} />
          <DetailList>
            <Detail label="Confidence" value={confidenceText(candidate.confidence)} />
            <Detail label="Method" value={candidate.method} />
          </DetailList>
          {candidate.sources?.map((source, sourceIndex) => (
            <EvidenceReference
              key={`contradiction-source-${sourceIndex}`}
              label={`Evidence source ${sourceIndex + 1}`}
              source={source}
            />
          ))}
        </FindingCard>
      ))}

      {result.stage_status && Object.keys(result.stage_status).length > 0 && (
        <div className="stage-status" aria-label="Analysis stage statuses">
          <h4>Processing stages</h4>
          <div className="status-row">
            {Object.entries(result.stage_status).map(([stage, status]) => (
              <StatusBadge key={stage} label={stage} value={status} />
            ))}
          </div>
        </div>
      )}
      <TextList
        label="Analysis warnings"
        values={result.warnings?.map((warning) => `${warning.code}: ${warning.message}`)}
      />
      <TextList
        label="Analysis errors"
        values={result.errors?.map((error) => `${error.code}: ${error.message}`)}
      />
    </article>
  );
}

export default function AnalysisFindingsSection({ analysis }) {
  const results = analysis?.results ?? [];
  return (
    <section className="report-section" aria-labelledby="analysis-title">
      <div className="section-heading section-heading--with-status">
        <div>
          <p className="section-kicker">Supplied evidence analysis</p>
          <h2 id="analysis-title">Analysis findings</h2>
        </div>
        <StatusBadge label="Availability" value={analysis?.availability} />
      </div>
      {!analysis?.supplied ? (
        <p className="empty-state">Analysis results were not supplied.</p>
      ) : results.length === 0 ? (
        <p className="empty-state">Analysis was supplied, but no results were recorded.</p>
      ) : (
        <div className="analysis-list">
          {results.map((result, index) => (
            <AnalysisDocument key={`${result.document_id}-${index}`} result={result} index={index} />
          ))}
        </div>
      )}
    </section>
  );
}
