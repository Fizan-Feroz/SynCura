// Single source of truth for the backend base URL.
// Set VITE_API_URL in .env to point at a remote backend; defaults to local dev.
export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// Training endpoints accept an optional token (backend TRAINING_API_TOKEN).
// Sent only when configured; the internal training UI is hidden by default.
export function trainingHeaders() {
  const token = import.meta.env.VITE_TRAINING_API_TOKEN
  return token ? { 'X-Training-Token': token } : {}
}

export async function apiGet(path) {
  const res = await fetch(`${API_URL}${path}`)
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`)
  return res.json()
}

export async function apiPost(path, body, extraHeaders = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = `POST ${path} -> ${res.status}`
    try {
      const data = await res.json()
      if (data && data.detail) detail = data.detail
    } catch {
      /* keep status text */
    }
    throw new Error(detail)
  }
  return res.json()
}
