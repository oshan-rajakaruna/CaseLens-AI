const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function getBackendHealth() {
  const response = await fetch(API_BASE_URL + "/health");

  if (!response.ok) {
    throw new Error("CaseLens backend health request failed.");
  }

  return response.json();
}
