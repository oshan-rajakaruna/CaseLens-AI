const plannedCapabilities = [
  "Evidence-grounded research reports",
  "Citation and provenance review",
  "Human-reviewed legal research",
];

export default function HomePage() {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <p className="eyebrow">Academic research project</p>
      <h1 id="hero-title">CaseLens</h1>
      <p className="subtitle">
        AI-Powered Legal Case Analysis, Evidence Review and Precedent Retrieval
        Assistant
      </p>

      <div className="status-card">
        <p className="status-label">Current application status</p>
        <h2>Research workspace in development</h2>
        <p>
          Backend summarization, verification and structured report components exist.
          Secure case APIs, saved reports, authentication and browser workflow
          integration are still pending.
        </p>
      </div>

      <ul className="capability-list" aria-label="Planned capabilities">
        {plannedCapabilities.map((capability) => (
          <li key={capability}>{capability}</li>
        ))}
      </ul>
    </section>
  );
}
