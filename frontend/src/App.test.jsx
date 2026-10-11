import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { shouldShowLegalReportPreview } from "./App.jsx";
import ReportPreviewPage from "./pages/ReportPreviewPage.jsx";

describe("development report preview gate", () => {
  it("requires both development mode and the explicit query parameter", () => {
    expect(
      shouldShowLegalReportPreview({
        isDevelopment: true,
        search: "?preview=legal-report",
      }),
    ).toBe(true);
    expect(
      shouldShowLegalReportPreview({
        isDevelopment: true,
        search: "",
      }),
    ).toBe(false);
  });

  it("does not expose the preview in production", () => {
    expect(
      shouldShowLegalReportPreview({
        isDevelopment: false,
        search: "?preview=legal-report",
      }),
    ).toBe(false);
  });

  it("labels the preview as synthetic development data", () => {
    render(<ReportPreviewPage />);

    expect(
      screen.getByText("SYNTHETIC TEST DATA — DEVELOPMENT PREVIEW"),
    ).toBeVisible();
  });
});
