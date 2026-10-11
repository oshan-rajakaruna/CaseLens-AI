export default function HumanReviewNotice({ disclaimer }) {
  return (
    <aside className="human-review-notice" aria-labelledby="human-review-title">
      <p className="section-kicker">Mandatory limitation</p>
      <h2 id="human-review-title">Human review required</h2>
      <p>{disclaimer || "A human-review disclaimer was not supplied."}</p>
    </aside>
  );
}
