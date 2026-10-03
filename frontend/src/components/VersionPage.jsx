import React from 'react'
import { APP_VERSION } from '../config'

// Release history sourced from the shipped git history (frontend/package.json
// version at each release commit). Newest first. Add a new entry at the top
// on every website release so /version stays the record of what was pushed.
const RELEASES = [
  {
    version: '1.12.1',
    date: '2026-10-04',
    kind: 'patch',
    title: 'Hidden version history page',
    notes: ['New /version route lists every website release with a short fix note.'],
  },
  {
    version: '1.12.0',
    date: '2026-09-30',
    kind: 'minor',
    title: 'Batched simulation ticks with timing',
    notes: ['Server tick scores beds in one batch and exposes per-tick timing.'],
  },
  {
    version: '1.11.0',
    date: '2026-09-29',
    kind: 'minor',
    title: 'Real brand marks and web analytics',
    notes: ['Theme-aware SynCura logo and favicon.', 'Vercel Web Analytics installed.'],
  },
  {
    version: '1.10.0',
    date: '2026-09-29',
    kind: 'minor',
    title: 'Model attribution on every alert',
    notes: ['Alert cards now show what the model saw behind each call.'],
  },
  {
    version: '1.9.1',
    date: '2026-09-29',
    kind: 'patch',
    title: 'Bed profile crash fix',
    notes: ['Fixed a temporal-dead-zone crash by declaring the bed before the effect.'],
  },
  {
    version: '1.9.0',
    date: '2026-09-29',
    kind: 'minor',
    title: 'Full-code audit fixes',
    notes: ['Audit corrections across ML, backend, frontend, and docs.'],
  },
  {
    version: '1.8.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Local-only demo source',
    notes: ['Demo engine runs entirely in this browser; nothing is synced or sent.'],
  },
  {
    version: '1.7.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Visible simulated scenario overlays',
    notes: ['Scenario nudges are labeled on screen; baseline stays pure model output.'],
  },
  {
    version: '1.6.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'One-click replay seeding',
    notes: ['Sample snapshot beds into the store when the Replay view is empty.'],
  },
  {
    version: '1.5.3',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Missing scenario label fix',
    notes: ['Defined the active scenario label used by the status pill.'],
  },
  {
    version: '1.5.2',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Replay hero film on every visit',
    notes: ['Landing film now replays on each home visit instead of once.'],
  },
  {
    version: '1.5.1',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Shared-engine controls in live mode',
    notes: ['Simulation controls render correctly for the shared backend engine.'],
  },
  {
    version: '1.5.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Backend-only beds for everyone',
    notes: ['Removed the old simulated mode; all browsers read beds from the backend.'],
  },
  {
    version: '1.4.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Replay source with real ingested data',
    notes: ['Replay beds are read back from backend-ingested clinical data.'],
  },
  {
    version: '1.3.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Shared backend simulation',
    notes: ['One server engine serves identical live beds to every browser, plus bed profiles.'],
  },
  {
    version: '1.2.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'One alert per bed',
    notes: ['The latest severity replaces older alerts instead of stacking.'],
  },
  {
    version: '1.1.0',
    date: '2026-09-28',
    kind: 'minor',
    title: 'Admin throughput panel',
    notes: ['Hidden /admin gained a live ingest-rate sparkline.'],
  },
  {
    version: '1.0.0',
    date: '2026-09-28',
    kind: 'major',
    title: 'First release with clinical alert terms',
    notes: ['Website v1.0.0; alert language aligned to clinical terms.'],
  },
  {
    version: '0.0.4',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Version bump',
    notes: ['Release bookkeeping.'],
  },
  {
    version: '0.0.3',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Hidden admin status panel',
    notes: ['New /admin route with live uptime, latency, and hosting detail.'],
  },
  {
    version: '0.0.2',
    date: '2026-09-28',
    kind: 'patch',
    title: 'Version tracking and training gate',
    notes: ['Backend reports the website version; training UI hidden from public builds.'],
  },
  {
    version: '0.0.1',
    date: '2026-09-25',
    kind: 'minor',
    title: 'ICU instruments redesign',
    notes: ['Chart-paper light and bedside-monitor dark themes across the site.'],
  },
]

const KIND_CLASS = { major: 'job-critical', minor: 'job-stable', patch: 'job-info' }

export default function VersionPage() {
  return (
    <div className="page page-narrow">
      <header className="page-head">
        <div>
          <p className="muted small">Release notes · research prototype</p>
          <h1>What changed <span className="num">v{APP_VERSION}</span></h1>
          <p className="lede">
            Every website update pushed to production, newest first — version, date,
            and a short note on what was fixed or added.
          </p>
        </div>
      </header>

      <ol className="versions">
        {RELEASES.map((r) => (
          <li key={r.version} className="version">
            <div className="version-head">
              <code className="num">v{r.version}</code>
              <span className={`job-status ${KIND_CLASS[r.kind]}`}>{r.kind}</span>
              <span className="muted small num">{r.date}</span>
              {r.version === APP_VERSION && <span className="job-status job-stable">current</span>}
            </div>
            <p className="version-title">{r.title}</p>
            <ul className="version-notes">
              {r.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  )
}
