import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import AppHeader from "../components/AppHeader.jsx";
import { syntheticLegalReport } from "../fixtures/syntheticLegalReport.js";
import CaseDetailPage from "./CaseDetailPage.jsx";
import CasesPage from "./CasesPage.jsx";
import DashboardPage from "./DashboardPage.jsx";
import HomePage from "./HomePage.jsx";
import ReportPage from "./ReportPage.jsx";
import ReportsPage from "./ReportsPage.jsx";

describe("Milestone 5B workspace pages", () => {
  it("renders the dashboard without invented statistics", () => {
    render(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "Dashboard" })).toBeVisible();
    expect(screen.queryByText(/total cases/i)).toBeNull();
    expect(screen.queryByText(/completed reports/i)).toBeNull();
    expect(screen.queryByText(/^0$/)).toBeNull();
  });

  it("identifies unavailable dashboard integrations accurately", () => {
    render(<DashboardPage />);

    expect(screen.getByText(/Case statistics, recent reports, active tasks/)).toBeVisible();
    expect(screen.getAllByText("Unavailable integration")).toHaveLength(3);
    expect(screen.getByText(/cannot yet create cases/)).toBeVisible();
  });

  it("distinguishes unavailable cases from a successful empty response", () => {
    const { rerender } = render(<CasesPage />);
    expect(screen.getByText(/No case list has been fetched/)).toBeVisible();
    expect(screen.queryByText(/returned no case records/)).toBeNull();

    rerender(<CasesPage cases={[]} state="ready" />);
    expect(screen.getByText(/completed case request returned no case records/)).toBeVisible();
  });

  it("renders only supplied case records and supports selection", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const caseRecord = {
      id: "CASE-REAL-SHAPED-001",
      title: "Supplied case title",
      description: "Supplied case description.",
    };
    render(<CasesPage cases={[caseRecord]} state="ready" onSelectCase={onSelect} />);

    expect(screen.getByText(caseRecord.id)).toBeVisible();
    expect(screen.getByText(caseRecord.title)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "View case" }));
    expect(onSelect).toHaveBeenCalledWith(caseRecord);
  });

  it("shows an honest case-detail state without a selected case", () => {
    render(<CaseDetailPage />);

    expect(screen.getByText(/No validated case record was selected/)).toBeVisible();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("renders supplied case details without inventing fields or actions", () => {
    render(<CaseDetailPage caseRecord={{ id: "CASE-002", title: "Supplied title" }} />);

    expect(screen.getByText("CASE-002")).toBeVisible();
    expect(screen.getByText("Supplied title")).toBeVisible();
    expect(screen.getByText("No case description was supplied.")).toBeVisible();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("supports unavailable, loading and error report states", () => {
    const { rerender } = render(<ReportsPage />);
    expect(screen.getByText(/No report list has been fetched/)).toBeVisible();

    rerender(<ReportsPage state="loading" />);
    expect(screen.getByText("Loading reports from the configured service.")).toBeVisible();

    rerender(<ReportsPage state="error" errorMessage="Sanitized report request failure." />);
    expect(screen.getByRole("alert")).toHaveTextContent("Sanitized report request failure.");
  });

  it("renders an empty report state only after a successful request", () => {
    render(<ReportsPage reports={[]} state="ready" />);

    expect(screen.getByText(/completed report request returned no report records/)).toBeVisible();
  });

  it("renders supplied report records and invokes selection", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const reportRecord = {
      id: "REPORT-001",
      title: "Supplied research report",
      case_id: "CASE-002",
    };
    render(
      <ReportsPage
        reports={[reportRecord]}
        state="ready"
        onSelectReport={onSelect}
      />,
    );

    expect(screen.getByText("REPORT-001")).toBeVisible();
    expect(screen.getByText("Case ID: CASE-002")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Open report" }));
    expect(onSelect).toHaveBeenCalledWith(reportRecord);
  });

  it("does not render report content without supplied data", () => {
    render(<ReportPage />);

    expect(screen.getByText(/No validated report was selected/)).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Legal Research Report" })).toBeNull();
  });

  it("reuses LegalReportView for a supplied structured report", () => {
    render(<ReportPage report={syntheticLegalReport} />);

    expect(screen.getByRole("heading", { name: "Legal Research Report" })).toBeVisible();
    expect(screen.getByText("Fictional Tenant v. Fictional Property Company")).toBeVisible();
    expect(screen.getByLabelText("Comparison verification: not_verified")).toBeVisible();
  });

  it("provides responsive semantic navigation and active-page indication", () => {
    render(<AppHeader activePage="cases" />);

    const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
    expect(navigation).toBeVisible();
    expect(screen.getAllByRole("link")).toHaveLength(5);
    expect(screen.getByRole("link", { name: "Cases" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("supports keyboard navigation through workspace links", async () => {
    const user = userEvent.setup();
    render(<AppHeader activePage="home" />);

    await user.tab();
    expect(screen.getByRole("link", { name: "CaseLens home" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("link", { name: "Home" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("link", { name: "Dashboard" })).toHaveFocus();
  });

  it("preserves the existing home page with accurate current copy", () => {
    render(<HomePage />);

    expect(screen.getByRole("heading", { name: "CaseLens" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Research workspace in development" })).toBeVisible();
    expect(screen.getByText(/Secure case APIs, saved reports/)).toBeVisible();
  });

  it("keeps synthetic fixtures out of normal workspace screens", () => {
    const { rerender } = render(<DashboardPage />);
    expect(screen.queryByText("Fictional Tenant v. Fictional Property Company")).toBeNull();
    rerender(<CasesPage />);
    expect(screen.queryByText("SYNTHETIC-CASE-001")).toBeNull();
    rerender(<ReportsPage />);
    expect(screen.queryByText("SYNTHETIC-CASE-001")).toBeNull();
  });

  it("makes no backend, Gemini or external request", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    render(<DashboardPage />);
    render(<CasesPage />);
    render(<ReportsPage />);
    render(<ReportPage />);

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
