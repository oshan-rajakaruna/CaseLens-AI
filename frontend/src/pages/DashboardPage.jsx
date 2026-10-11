const workspaceAreas = [
  {
    title: "Cases",
    description: "Case records and associated documents require the case-management backend.",
  },
  {
    title: "Research reports",
    description: "Report listing and retrieval require secure report APIs and persistence.",
  },
  {
    title: "Research workflow",
    description: "Coordinator execution and task statuses are not exposed to the browser.",
  },
];

export default function DashboardPage() {
  return (
    <div className="workspace-page" data-page="dashboard">
      <header className="workspace-page__header">
        <p className="eyebrow">Research workspace</p>
        <h1>Dashboard</h1>
        <p>
          A frontend foundation for future authenticated case research. Live workspace data
          is not connected in this milestone.
        </p>
      </header>

      <section className="workspace-section" aria-labelledby="workspace-overview-title">
        <h2 id="workspace-overview-title">Workspace overview</h2>
        <p className="integration-notice">
          Case statistics, recent reports, active tasks, and agent activity are unavailable
          because authenticated application APIs and persistence are not implemented.
        </p>
        <div className="workspace-card-grid">
          {workspaceAreas.map((area) => (
            <article className="workspace-card" key={area.title}>
              <p className="availability-label">Unavailable integration</p>
              <h3>{area.title}</h3>
              <p>{area.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="workspace-section" aria-labelledby="development-status-title">
        <h2 id="development-status-title">Development status</h2>
        <p>
          Evidence-grounded report components exist, but normal browser users cannot yet
          create cases, execute research workflows, or retrieve saved reports.
        </p>
      </section>
    </div>
  );
}
