const plannedCapabilities = [
  "Legal case analysis",
  "Evidence review",
  "Precedent retrieval",
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
        <p className="status-label">Current development phase</p>
        <h2>Project foundation</h2>
        <p>
          The application shell and system boundaries are in place. AI and
          legal-analysis capabilities are planned for later phases.
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
