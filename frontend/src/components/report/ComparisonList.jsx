import { safeHttpUrl } from "./AuthorityList.jsx";
import {
  Detail,
  DetailList,
  EvidenceReference,
  joined,
  TextList,
} from "./ReportPrimitives.jsx";
import StatusBadge from "./StatusBadge.jsx";

export default function ComparisonList({ comparisons = [], comparisonStatus }) {
  return (
    <section className="report-section" aria-labelledby="comparisons-title">
      <div className="section-heading section-heading--with-status">
        <div>
          <p className="section-kicker">Explicit source associations</p>
          <h2 id="comparisons-title">Fact-to-authority links</h2>
        </div>
        <StatusBadge label="Comparison status" value={comparisonStatus} />
      </div>
      {comparisons.length === 0 ? (
        <p className="empty-state">
          Substantive legal comparison was not assessed and no explicit links were recorded.
        </p>
      ) : (
        <div className="comparison-list">
          {comparisons.map((comparison) => (
            <article className="report-card comparison-card" key={comparison.id}>
              <div className="card-heading card-heading--with-status">
                <div>
                  <span className="reference-tag">{comparison.id}</span>
                  <h3>Selected fact and authority</h3>
                </div>
                <StatusBadge label="Review status" value={comparison.review_status} />
              </div>

              <div className="comparison-columns">
                <div>
                  <h4>Selected fact</h4>
                  <p className="finding-text">{comparison.fact?.statement}</p>
                  <div className="status-row">
                    <StatusBadge
                      label="Epistemic status"
                      value={comparison.fact?.epistemic_status}
                    />
                    <StatusBadge
                      label="Fact verification"
                      value={comparison.fact?.verification_status}
                    />
                  </div>
                  <EvidenceReference source={comparison.fact?.source} />
                </div>
                <div>
                  <h4>Selected authority</h4>
                  <DetailList>
                    <Detail label="CTX reference" value={comparison.authority?.context_id} />
                    <Detail label="Title" value={comparison.authority?.title} />
                    <Detail label="Document ID" value={comparison.authority?.document_id} />
                    <Detail
                      label="Official citation"
                      value={comparison.authority?.official_citation}
                    />
                    <Detail label="Court" value={comparison.authority?.court} />
                    <Detail label="Source" value={comparison.authority?.source} />
                    <Detail label="Pages" value={joined(comparison.authority?.pages)} />
                    <Detail
                      label="Chunk IDs"
                      value={joined(comparison.authority?.chunk_ids)}
                    />
                    <Detail label="Provenance" value={comparison.authority?.provenance} />
                  </DetailList>
                  {safeHttpUrl(comparison.authority?.source_url) && (
                    <a
                      className="source-link"
                      href={safeHttpUrl(comparison.authority.source_url)}
                      rel="noreferrer"
                      target="_blank"
                    >
                      Open selected authority source
                    </a>
                  )}
                </div>
              </div>

              <div className="relationship-note">
                <h4>Recorded relationship</h4>
                <p>{comparison.relationship}</p>
              </div>
              <TextList label="Link limitations" values={comparison.limitations} />

              {comparison.ai_draft && (
                <aside className="ai-draft" aria-label="Unverified AI comparison draft">
                  <p className="ai-draft__label">
                    Not independently verified — Human review required
                  </p>
                  <div className="status-row">
                    <StatusBadge
                      label="Comparison verification"
                      value={comparison.ai_draft.verification_status}
                    />
                    <StatusBadge
                      label="Human review required"
                      value={comparison.ai_draft.human_review_required ? "yes" : "no"}
                    />
                  </div>
                  <h4>AI-generated comparative explanation</h4>
                  <p className="long-text">{comparison.ai_draft.explanation}</p>
                  <TextList
                    label="Uncertainties and limitations"
                    values={comparison.ai_draft.uncertainties}
                  />
                  <p className="provider-note">
                    Provider: {comparison.ai_draft.provider}
                    {comparison.ai_draft.model ? ` / ${comparison.ai_draft.model}` : ""}
                  </p>
                </aside>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
