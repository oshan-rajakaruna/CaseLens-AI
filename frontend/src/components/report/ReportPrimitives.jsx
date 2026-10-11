export function DetailList({ children }) {
  return <dl className="detail-list">{children}</dl>;
}

export function Detail({ label, value }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="detail-list__item">
      <dt>{label}</dt>
      <dd>{String(value)}</dd>
    </div>
  );
}

export function TextList({ label, values }) {
  if (!Array.isArray(values) || values.length === 0) return null;
  return (
    <div className="text-list">
      <h4>{label}</h4>
      <ul>
        {values.map((value, index) => (
          <li key={`${label}-${index}`}>{String(value)}</li>
        ))}
      </ul>
    </div>
  );
}

export function EvidenceReference({ source, label = "Evidence source" }) {
  if (!source) return null;
  return (
    <div className="evidence-reference" aria-label={label}>
      <strong>{label}</strong>
      <span>Document: {source.document_id ?? "unavailable"}</span>
      {source.page != null && <span>Page: {source.page}</span>}
      {source.paragraph != null && <span>Paragraph: {source.paragraph}</span>}
      {source.locator && <span>Locator: {source.locator}</span>}
    </div>
  );
}

export function confidenceText(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : null;
}

export function joined(values) {
  return Array.isArray(values) && values.length ? values.join(", ") : null;
}
