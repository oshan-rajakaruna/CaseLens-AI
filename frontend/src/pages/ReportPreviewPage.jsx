import LegalReportView from "../components/report/LegalReportView.jsx";
import { syntheticLegalReport } from "../fixtures/syntheticLegalReport.js";

export default function ReportPreviewPage() {
  return (
    <div className="preview-page">
      <div className="development-banner" role="status">
        SYNTHETIC TEST DATA — DEVELOPMENT PREVIEW
      </div>
      <LegalReportView report={syntheticLegalReport} />
    </div>
  );
}
