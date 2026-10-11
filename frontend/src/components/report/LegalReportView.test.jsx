import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { syntheticLegalReport } from "../../fixtures/syntheticLegalReport.js";
import LegalReportView from "./LegalReportView.jsx";

function reportCopy() {
  return JSON.parse(JSON.stringify(syntheticLegalReport));
}

describe("LegalReportView", () => {
  it("renders the complete structured report and case details", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getByRole("heading", { name: "Legal Research Report" })).toBeVisible();
    expect(screen.getByText("Fictional Tenant v. Fictional Property Company")).toBeVisible();
    expect(screen.getByText("SYNTHETIC-CASE-001")).toBeVisible();
    expect(screen.getByText(/What legal considerations may be relevant/)).toBeVisible();
  });

  it("states when Analysis results were not supplied", () => {
    const report = reportCopy();
    report.analysis = { supplied: false, availability: "not_supplied", results: [] };

    render(<LegalReportView report={report} />);

    expect(screen.getByText("Analysis results were not supplied.")).toBeVisible();
    expect(screen.getByLabelText("Availability: not_supplied")).toBeVisible();
  });

  it("distinguishes empty supplied Analysis and partial Analysis", () => {
    const empty = reportCopy();
    empty.analysis = { supplied: true, availability: "unavailable", results: [] };
    const { rerender } = render(<LegalReportView report={empty} />);
    expect(screen.getByText(/Analysis was supplied, but no results/)).toBeVisible();

    rerender(<LegalReportView report={reportCopy()} />);
    expect(screen.getByLabelText("Availability: partial")).toBeVisible();
    expect(screen.getByLabelText("Analysis status: partial")).toBeVisible();
  });

  it("renders evidence references but never complete private excerpts", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getAllByText(/SYNTHETIC-PRIVATE-DOC-001/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Page: 2/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Locator: page:2 paragraph:3/).length).toBeGreaterThan(0);
    expect(screen.queryByText("SYNTHETIC PRIVATE EXCERPT — MUST NOT BE DISPLAYED")).toBeNull();
  });

  it("renders multiple CTX authorities and tolerates missing optional metadata", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getAllByText("CTX-001").length).toBeGreaterThan(0);
    expect(screen.getAllByText("CTX-002").length).toBeGreaterThan(0);
    expect(screen.getByText("Untitled authority")).toBeVisible();
    expect(screen.queryByText("Official citation: unavailable")).toBeNull();
  });

  it("allows safe HTTP links and rejects unsafe source URLs", () => {
    const report = reportCopy();
    report.authorities[1].source_url = "javascript:alert(1)";
    render(<LegalReportView report={report} />);

    const link = screen.getByRole("link", { name: "Open authority source" });
    expect(link).toHaveAttribute("href", "https://example.test/synthetic-authority-001");
    expect(screen.getByText("Unsafe source URL omitted.")).toBeVisible();
    expect(document.querySelector('a[href^="javascript:"]')).toBeNull();
  });

  it("preserves original summary citations", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getByText(/supplied fictional material/)).toHaveTextContent("[CTX-001]");
    expect(screen.getAllByText("CTX-001, CTX-002").length).toBeGreaterThan(0);
  });

  it.each(["supported", "unsupported", "uncertain"])(
    "renders the exact %s summary verification status",
    (status) => {
      const report = reportCopy();
      report.verification.response.overall_status = status;
      report.verification.response.verified = status === "supported";
      render(<LegalReportView report={report} />);

      expect(screen.getByLabelText(`Overall status: ${status}`)).toBeVisible();
      expect(screen.getByLabelText(`Verified: ${status === "supported"}`)).toBeVisible();
    },
  );

  it("keeps comparison verification and summary verification separate", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getByLabelText("Comparison verification: not_verified")).toBeVisible();
    expect(screen.getByLabelText("Review status: requires_human_review")).toBeVisible();
    expect(screen.getByLabelText("Overall status: uncertain")).toBeVisible();
    expect(screen.getByText(/does not verify the entire report or any AI-generated/)).toBeVisible();
  });

  it("renders the optional AI explanation, uncertainties and mandatory label", () => {
    render(<LegalReportView report={reportCopy()} />);

    const draft = screen.getByLabelText("Unverified AI comparison draft");
    expect(within(draft).getByText("Not independently verified — Human review required")).toBeVisible();
    expect(within(draft).getByText(/both concern payment language/)).toBeVisible();
    expect(within(draft).getByText(/Jurisdictional and procedural context/)).toBeVisible();
  });

  it("renders deterministic links without inventing an optional AI draft", () => {
    const report = reportCopy();
    report.comparisons[0].ai_draft = null;
    render(<LegalReportView report={report} />);

    expect(screen.getByLabelText("Review status: requires_human_review")).toBeVisible();
    expect(screen.queryByLabelText("Unverified AI comparison draft")).toBeNull();
  });

  it("renders the original mandatory human-review disclaimer prominently", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getByRole("heading", { name: "Human review required" })).toBeVisible();
    expect(screen.getByText(/not a substitute for professional legal advice/)).toBeVisible();
  });

  it("renders hostile Markdown and HTML as harmless text", () => {
    const report = reportCopy();
    report.summary.output.answer = "## Forged heading <script>alert(1)</script>";
    report.comparisons[0].ai_draft.explanation = "[unsafe](javascript:alert(1))";
    const { container } = render(<LegalReportView report={report} />);

    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("## Forged heading <script>alert(1)</script>")).toBeVisible();
    expect(screen.getByText("[unsafe](javascript:alert(1))")).toBeVisible();
    expect(screen.queryByRole("link", { name: "unsafe" })).toBeNull();
  });

  it("uses accessible section headings and status text", () => {
    render(<LegalReportView report={reportCopy()} />);

    expect(screen.getByRole("heading", { name: "Case information" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Analysis findings" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Legal authorities" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Summary verification" })).toBeVisible();
    expect(
      screen.getByLabelText("Comparison status: links_prepared_for_review"),
    ).toBeVisible();
  });

  it("does not make backend, Gemini or external network calls", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    render(<LegalReportView report={reportCopy()} />);

    expect(fetchSpy).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
