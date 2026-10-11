const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function getBackendHealth() {
  const response = await fetch(API_BASE_URL + "/health");

  if (!response.ok) {
    throw new Error("CaseLens backend health request failed.");
  }

  return response.json();
}

function token() { return localStorage.getItem("caselens_access_token"); }
async function request(path, options = {}) {
  const response = await fetch(API_BASE_URL + path, { ...options, headers: { "Content-Type": "application/json", ...(token() ? { Authorization: `Bearer ${token()}` } : {}), ...options.headers } });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || "Request could not be completed."); }
  return response.status === 204 ? null : response.json();
}
export const getCases = () => request("/api/cases");
export const createCase = (body) => request("/api/cases", { method: "POST", body: JSON.stringify(body) });
export const searchPrecedents = (body) => request("/api/retrieval/search", { method: "POST", body: JSON.stringify(body) });
