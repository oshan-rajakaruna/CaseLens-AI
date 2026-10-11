import { Detail, DetailList, joined } from "./ReportPrimitives.jsx";

export function safeHttpUrl(value) {
  if (typeof value !== "string" || !value) return null;
  if (/\s|[\u0000-\u001f<>"\\]/u.test(value)) return null;
  try {
    const parsed = new URL(value);
    if (!["http:", "https:"].includes(parsed.protocol)) return null;
    if (!parsed.hostname || parsed.username || parsed.password) return null;
    return parsed.href;
  } catch {
    return null;
  }
}

export default function AuthorityList({ authorities = [] }) {
  return (
    <section className="report-section" aria-labelledby="authorities-title">
      <div className="section-heading">
        <p className="section-kicker">Retrieved sources</p>
        <h2 id="authorities-title">Legal authorities</h2>
      </div>
      {authorities.length === 0 ? (
        <p className="empty-state">No summary-cited authorities were recorded.</p>
      ) : (
        <div className="card-grid">
          {authorities.map((authority) => {
            const safeUrl = safeHttpUrl(authority.source_url);
            return (
              <article className="report-card authority-card" key={authority.context_id}>
                <div className="card-heading">
                  <span className="reference-tag">{authority.context_id}</span>
                  <h3>{authority.title ?? "Untitled authority"}</h3>
                </div>
                <DetailList>
                  <Detail label="Document ID" value={authority.document_id} />
                  <Detail label="Official citation" value={authority.official_citation} />
                  <Detail label="Court" value={authority.court} />
                  <Detail label="Source" value={authority.source} />
                  <Detail label="Pages" value={joined(authority.pages)} />
                  <Detail label="Chunk IDs" value={joined(authority.chunk_ids)} />
                  <Detail label="Provenance" value={authority.provenance} />
                </DetailList>
                {safeUrl ? (
                  <a className="source-link" href={safeUrl} rel="noreferrer" target="_blank">
                    Open authority source
                  </a>
                ) : authority.source_url ? (
                  <p className="unsafe-link-note">Unsafe source URL omitted.</p>
                ) : null}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
