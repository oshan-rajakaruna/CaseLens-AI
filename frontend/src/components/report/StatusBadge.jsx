function statusTone(status) {
  const normalized = String(status ?? "unavailable").toLowerCase();
  if (["supported", "completed", "valid"].includes(normalized)) return "positive";
  if (["failed", "unsupported", "invalid", "invalid_citations"].includes(normalized)) {
    return "critical";
  }
  if (
    normalized.includes("review") ||
    normalized.includes("uncertain") ||
    normalized.includes("partial") ||
    normalized.includes("not_verified")
  ) {
    return "caution";
  }
  return "neutral";
}

export default function StatusBadge({ label, value }) {
  const displayedValue = value ?? "unavailable";
  return (
    <span
      aria-label={`${label}: ${String(displayedValue)}`}
      className={`status-badge status-badge--${statusTone(displayedValue)}`}
    >
      <span className="status-badge__label">{label}:</span> {String(displayedValue)}
    </span>
  );
}
