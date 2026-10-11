import { Detail, DetailList, TextList } from "./ReportPrimitives.jsx";
import StatusBadge from "./StatusBadge.jsx";

export default function VerificationSection({ verification }) {
  const response = verification?.response;
  if (!response) {
    return (
      <section className="report-section" aria-labelledby="verification-title">
        <h2 id="verification-title">Summary verification</h2>
        <p className="empty-state">Verification findings are unavailable.</p>
      </section>
    );
  }

  return (
    <section className="report-section" aria-labelledby="verification-title">
      <div className="section-heading section-heading--with-status">
        <div>
          <p className="section-kicker">Generated summary check</p>
          <h2 id="verification-title">Summary verification</h2>
        </div>
        <div className="status-row">
          <StatusBadge label="Overall status" value={response.overall_status} />
          <StatusBadge label="Verified" value={response.verified ? "true" : "false"} />
        </div>
      </div>

      <div className="scope-notice">
        <strong>Verification scope</strong>
        <p>{verification.scope}</p>
        <p>
          This status applies to the generated summary claim. It does not verify the entire
          report or any AI-generated comparison draft.
        </p>
      </div>

      {response.claim_results?.length ? (
        <div className="claim-list">
          {response.claim_results.map((result, index) => (
            <article className="finding-card" key={`${result.claim?.id}-${index}`}>
              <div className="card-heading card-heading--with-status">
                <h3>Claim {index + 1}</h3>
                <StatusBadge label="Claim status" value={result.status} />
              </div>
              <p className="finding-text">{result.claim?.text}</p>
              <DetailList>
                <Detail label="Claim ID" value={result.claim?.id} />
                <Detail label="Evidence IDs" value={result.evidence_ids?.join(", ")} />
              </DetailList>
              {result.citation_results?.map((citation, citationIndex) => (
                <div className="citation-result" key={`citation-${citationIndex}`}>
                  <StatusBadge label="Citation status" value={citation.status} />
                  <DetailList>
                    <Detail
                      label="Citation document"
                      value={citation.citation?.document_id}
                    />
                    <Detail label="Citation locator" value={citation.citation?.locator} />
                    <Detail label="Matched evidence" value={citation.evidence_id} />
                  </DetailList>
                  <TextList label="Citation issues" values={citation.issues} />
                </div>
              ))}
              <TextList label="Claim issues" values={result.issues} />
            </article>
          ))}
        </div>
      ) : (
        <p className="empty-state">No claim-level verification findings were recorded.</p>
      )}

      <TextList label="Verification notes" values={response.notes} />
      <TextList label="Bridge warnings" values={verification.bridge_warnings} />
      {response.warnings?.length > 0 && (
        <div className="text-list">
          <h3>Responsible AI warnings</h3>
          <ul>
            {response.warnings.map((warning, index) => (
              <li key={`${warning.code}-${index}`}>
                <strong>{warning.severity}:</strong> {warning.message} ({warning.code})
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
