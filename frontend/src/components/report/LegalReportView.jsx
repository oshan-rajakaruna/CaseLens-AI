import AnalysisFindingsSection from "./AnalysisFindingsSection.jsx";
import AuthorityList from "./AuthorityList.jsx";
import CaseInformationSection from "./CaseInformationSection.jsx";
import ComparisonList from "./ComparisonList.jsx";
import HumanReviewNotice from "./HumanReviewNotice.jsx";
import { Detail, DetailList, TextList } from "./ReportPrimitives.jsx";
import VerificationSection from "./VerificationSection.jsx";

function SummarySection({ summary }) {
  if (!summary?.output) {
    return (
      <section className="report-section" aria-labelledby="summary-title">
        <h2 id="summary-title">Generated legal summary</h2>
        <p className="empty-state">A generated summary is unavailable.</p>
      </section>
    );
  }
  return (
    <section className="report-section" aria-labelledby="summary-title">
      <div className="section-heading">
        <p className="section-kicker">Evidence-grounded draft</p>
        <h2 id="summary-title">Generated legal summary</h2>
      </div>
      <p className="long-text summary-answer">{summary.output.answer}</p>
      <DetailList>
        <Detail label="Used CTX citations" value={summary.used_context_ids?.join(", ")} />
        <Detail label="Declared citations" value={summary.output.citations?.join(", ")} />
        <Detail label="Confidence" value={summary.output.confidence} />
        <Detail label="Provider" value={summary.provider_metadata?.provider} />
        <Detail label="Model" value={summary.provider_metadata?.model} />
      </DetailList>
      <TextList label="Summary limitations" values={summary.output.limitations} />
      <TextList label="Summary warnings" values={summary.warnings} />
    </section>
  );
}

function DiagnosticsSection({ diagnostics }) {
  if (!diagnostics) return null;
  return (
    <section className="report-section report-section--compact" aria-labelledby="diagnostics-title">
      <div className="section-heading">
        <p className="section-kicker">Retrieval context</p>
        <h2 id="diagnostics-title">Research diagnostics</h2>
      </div>
      <DetailList>
        <Detail label="Retrieval results" value={diagnostics.retrieval_result_count} />
        <Detail label="Context passages" value={diagnostics.rag_passage_count} />
        <Detail label="Source documents" value={diagnostics.source_document_count} />
        <Detail label="Context truncated" value={String(diagnostics.context_truncated)} />
        <Detail label="Weak context IDs" value={diagnostics.weak_context_ids?.join(", ")} />
      </DetailList>
      <TextList label="Retrieval warnings" values={diagnostics.warnings} />
    </section>
  );
}

export default function LegalReportView({ report }) {
  if (!report) {
    return <p className="empty-state">No legal research report was supplied.</p>;
  }

  return (
    <article className="legal-report" aria-labelledby="legal-report-title">
      <header className="report-hero">
        <p className="eyebrow">CaseLens research draft</p>
        <h1 id="legal-report-title">Legal Research Report</h1>
        <p>
          Structured research material for professional review. Citation identity and
          automated verification do not establish legal correctness.
        </p>
      </header>

      <HumanReviewNotice disclaimer={report.human_review_disclaimer} />
      <CaseInformationSection information={report.case_information} />
      <AnalysisFindingsSection analysis={report.analysis} />
      <AuthorityList authorities={report.authorities} />
      <ComparisonList
        comparisons={report.comparisons}
        comparisonStatus={report.legal_comparison_status}
      />
      <SummarySection summary={report.summary} />
      <VerificationSection verification={report.verification} />
      <DiagnosticsSection diagnostics={report.retrieval_diagnostics} />

      <section className="report-section" aria-labelledby="limitations-title">
        <div className="section-heading">
          <p className="section-kicker">Use constraints</p>
          <h2 id="limitations-title">Report limitations</h2>
        </div>
        {report.limitations?.length ? (
          <ul className="limitations-list">
            {report.limitations.map((limitation, index) => (
              <li key={`limitation-${index}`}>{limitation}</li>
            ))}
          </ul>
        ) : (
          <p className="empty-state">No additional report limitations were recorded.</p>
        )}
      </section>
    </article>
  );
}
