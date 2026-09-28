"""Self-contained HTML status page for the SynCura backend.

Served at ``/``. Purely informational: it renders the payloads of the existing
JSON endpoints so the page never reports a number the API does not expose.
No new dependencies - the page is plain HTML/CSS/JS and talks to the same
origin it was served from, so it works behind Render without CORS changes.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SynCura Backend Status</title>
<style>
  :root {
    --bg: #0b0f14;
    --panel: #131a22;
    --line: #24303c;
    --text: #e6edf3;
    --muted: #8b98a5;
    --ok: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
    --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 2rem 1.25rem 3rem;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.5 ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  main { max-width: 940px; margin: 0 auto; }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.75rem 1rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 1.4rem; margin: 0; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 0.85rem; }
  .pill {
    margin-left: auto;
    padding: 0.2rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .pill.ok { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 45%, transparent); }
  .pill.bad { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 45%, transparent); }
  section { margin-top: 2rem; }
  h2 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--muted);
    margin: 0 0 0.75rem;
    font-weight: 600;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.75rem;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.85rem 1rem;
  }
  .k { color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .v { margin-top: 0.3rem; font-size: 1.05rem; font-weight: 600; word-break: break-word; }
  .v.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.95rem; }
  .v.ok { color: var(--ok); }
  .v.warn { color: var(--warn); }
  .v.bad { color: var(--bad); }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .links { display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; font-size: 0.88rem; }
  footer { margin-top: 2.5rem; color: var(--muted); font-size: 0.78rem; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>SynCura Backend</h1>
      <div class="sub" id="service">service status</div>
    </div>
    <div class="pill" id="state">checking</div>
  </header>

  <section>
    <h2>Version</h2>
    <div class="grid">
      <div class="card"><div class="k">Website version</div><div class="v mono" id="website_version">-</div></div>
      <div class="card"><div class="k">Git commit</div><div class="v mono" id="git_commit">-</div></div>
      <div class="card"><div class="k">Python</div><div class="v mono" id="python_version">-</div></div>
      <div class="card"><div class="k">Render service</div><div class="v mono" id="render_service">-</div></div>
    </div>
  </section>

  <section>
    <h2>Model</h2>
    <div class="grid">
      <div class="card"><div class="k">Model id</div><div class="v mono" id="model_id">-</div></div>
      <div class="card"><div class="k">Validation AUC</div><div class="v" id="model_val_auc">-</div></div>
      <div class="card"><div class="k">Ensemble members</div><div class="v" id="ensemble_members">-</div></div>
      <div class="card"><div class="k">State</div><div class="v" id="model_state">-</div></div>
    </div>
  </section>

  <section>
    <h2>Runtime</h2>
    <div class="grid">
      <div class="card"><div class="k">Uptime</div><div class="v" id="uptime_human">-</div></div>
      <div class="card"><div class="k">Ingests total</div><div class="v" id="ingest_count">-</div></div>
      <div class="card"><div class="k">Ingests / min (1m)</div><div class="v" id="ingest_per_min_1m">-</div></div>
      <div class="card"><div class="k">Ingests / min (avg)</div><div class="v" id="ingest_per_min_avg">-</div></div>
    </div>
  </section>

  <section>
    <h2>Endpoints</h2>
    <div class="links" id="links"></div>
  </section>

  <footer>
    Research prototype. Not HIPAA-ready and not validated for clinical use.
    Values on this page are read live from <code>/version</code> and
    <code>/admin/status</code>.
  </footer>
</main>

<script>
const ENDPOINTS = [
  "/health", "/version", "/admin/status", "/metrics",
  "/patients", "/scores", "/simulation/state"
];

const set = (id, value) => {
  const el = document.getElementById(id);
  if (el) el.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
};

const num = (value, digits) =>
  typeof value === "number" ? value.toFixed(digits) : (value ?? "-");

async function getJSON(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

function render(version, admin) {
  set("service", (version.service || "syncura-backend") + " - " + (version.render_external_url || "local"));
  set("website_version", version.website_version);
  set("git_commit", version.git_commit);
  set("python_version", version.python_version);
  set("render_service", version.render_service);

  set("model_id", admin.model.model_id);
  set("model_val_auc", num(admin.model.model_val_auc, 3));
  set("ensemble_members", admin.model.ensemble_members);

  const degraded = Boolean(admin.model.degraded);
  const stateEl = document.getElementById("model_state");
  stateEl.textContent = degraded ? (admin.model.load_error || "degraded") : "loaded";
  stateEl.className = "v " + (degraded ? "bad" : "ok");

  set("uptime_human", admin.uptime_human);
  set("ingest_count", admin.runtime.ingest_count);
  set("ingest_per_min_1m", admin.runtime.ingest_per_min_1m);
  set("ingest_per_min_avg", admin.runtime.ingest_per_min_avg);

  const pill = document.getElementById("state");
  const healthy = admin.status === "ok" && !degraded;
  pill.textContent = healthy ? "operational" : admin.status;
  pill.className = "pill " + (healthy ? "ok" : "bad");
}

async function refresh() {
  try {
    const [version, admin] = await Promise.all([getJSON("/version"), getJSON("/admin/status")]);
    render(version, admin);
  } catch (err) {
    const pill = document.getElementById("state");
    pill.textContent = "unreachable";
    pill.className = "pill bad";
  }
}

document.getElementById("links").innerHTML = ENDPOINTS
  .map((p) => '<a href="' + p + '">' + p + "</a>")
  .join("");

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
