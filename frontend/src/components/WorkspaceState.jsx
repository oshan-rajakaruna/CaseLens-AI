const stateContent = {
  unavailable: {
    title: "Integration unavailable",
    defaultMessage: "This workspace requires a backend integration that is not available yet.",
  },
  loading: {
    title: "Loading",
    defaultMessage: "A real data request is in progress.",
  },
  empty: {
    title: "No records returned",
    defaultMessage: "The completed request returned no records.",
  },
  error: {
    title: "Request unsuccessful",
    defaultMessage: "The real data request could not be completed.",
  },
};

export default function WorkspaceState({ state, message }) {
  const content = stateContent[state] ?? stateContent.unavailable;
  return (
    <div
      className={`workspace-state workspace-state--${state}`}
      role={state === "error" ? "alert" : "status"}
    >
      <h2>{content.title}</h2>
      <p>{message ?? content.defaultMessage}</p>
    </div>
  );
}
