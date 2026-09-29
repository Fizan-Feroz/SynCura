import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { API_URL, apiGet } from '../api'
import { APP_VERSION } from '../config'
import { seriesPath } from './trace'

const POLL_MS = 5000
const HISTORY_LIMIT = 60
const EVENTS_KEY = 'syncura-admin-events'

function loadEvents() {
  try {
    const raw = localStorage.getItem(EVENTS_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.slice(0, 50) : []
  } catch {
    return []
  }
}

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return '—'
  }
}

function latencyTone(ms) {
  if (ms == null) return 'stable'
  if (ms >= 2000) return 'critical'
  if (ms >= 800) return 'high'
  if (ms >= 300) return 'watch'
  return 'stable'
}

export default function AdminPanel() {
  const [probe, setProbe] = useState({ checking: true, online: null, latencyMs: null, error: null, checkedAt: null })
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState(null)
  const [version, setVersion] = useState(null)
  const [history, setHistory] = useState([])
  const [events, setEvents] = useState(loadEvents)
  const timer = useRef(null)
  const lastOnline = useRef(null)

  const recordEvent = useCallback((online, detail) => {
    setEvents((prev) => {
      const entry = { t: Date.now(), online, detail: detail || (online ? 'Backend reachable' : 'Backend unreachable') }
      const next = [entry, ...prev].slice(0, 50)
      try {
        localStorage.setItem(EVENTS_KEY, JSON.stringify(next))
      } catch {
        /* storage unavailable */
      }
      return next
    })
  }, [])

  const check = useCallback(async () => {
    const started = performance.now()
    let sawSuccess = false
    try {
      let payload = null
      try {
        payload = await apiGet('/admin/status')
        sawSuccess = true
      } catch {
        payload = null
      }
      const latencyMs = Math.round(performance.now() - started)
      let healthPayload = null
      let versionPayload = null
      try {
        healthPayload = payload ? null : await apiGet('/health')
        if (!payload) sawSuccess = true
      } catch {
        healthPayload = null
      }
      try {
        versionPayload = await apiGet('/version')
        sawSuccess = true
      } catch {
        versionPayload = null
      }
      if (!sawSuccess) throw new Error('all probes failed')
      const checkedAt = Date.now()
      setProbe({ checking: false, online: true, latencyMs, error: null, checkedAt })
      setStatus(payload)
      setHealth(payload ? { status: payload.status, model_loaded: payload.model_loaded } : healthPayload)
      setVersion(versionPayload)
      setHistory((prev) => [...prev.slice(-(HISTORY_LIMIT - 1)), {
        t: checkedAt,
        latencyMs,
        online: true,
        ingests: typeof payload?.runtime?.ingest_count === 'number' ? payload.runtime.ingest_count : null,
      }])
      if (lastOnline.current === false) recordEvent(true, `Recovered in ${latencyMs} ms`)
      lastOnline.current = true
    } catch (error) {
      const checkedAt = Date.now()
      setProbe({ checking: false, online: false, latencyMs: null, error: error?.message || 'Unreachable', checkedAt })
      setHistory((prev) => [...prev.slice(-(HISTORY_LIMIT - 1)), { t: checkedAt, latencyMs: null, online: false }])
      if (lastOnline.current !== false) recordEvent(false, error?.message || 'Probe failed')
      lastOnline.current = false
    }
  }, [recordEvent])

  useEffect(() => {
    check()
    timer.current = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') check()
    }, POLL_MS)
    return () => window.clearInterval(timer.current)
  }, [check])

  const stats = useMemo(() => {
    const samples = history.filter((h) => h.online && typeof h.latencyMs === 'number').map((h) => h.latencyMs)
    const checks = history.length
    const outages = history.filter((h) => !h.online).length
    if (!samples.length) return { checks, outages, uptimePct: checks ? 0 : null, avg: null, min: null, max: null }
    const sum = samples.reduce((a, b) => a + b, 0)
    return {
      checks,
      outages,
      uptimePct: Math.round(((checks - outages) / checks) * 1000) / 10,
      avg: Math.round(sum / samples.length),
      min: Math.min(...samples),
      max: Math.max(...samples),
    }
  }, [history])

  const spark = useMemo(() => {
    const values = history.filter((h) => h.online && typeof h.latencyMs === 'number').map((h) => h.latencyMs)
    if (values.length < 2) return ''
    return seriesPath(values, 560, 120, 8)
  }, [history])

  const throughput = useMemo(() => {
    const rates = []
    for (let i = 1; i < history.length; i++) {
      const a = history[i - 1]
      const b = history[i]
      if (typeof a.ingests === 'number' && typeof b.ingests === 'number') {
        const dtMin = (b.t - a.t) / 60000
        // A counter drop means the backend restarted — clamp, don't go negative.
        if (dtMin > 0) rates.push(Math.max(0, (b.ingests - a.ingests) / dtMin))
      }
    }
    if (!rates.length) return { rates, latest: null, avg: null }
    const sum = rates.reduce((x, y) => x + y, 0)
    return { rates, latest: rates[rates.length - 1], avg: sum / rates.length }
  }, [history])

  const throughputSpark = useMemo(() => {
    if (throughput.rates.length < 2) return ''
    const max = Math.max(...throughput.rates, 1)
    return seriesPath(throughput.rates, 560, 120, 8, [0, max])
  }, [throughput])

  function formatAgo(tsSeconds) {
    if (tsSeconds == null) return '—'
    const secs = Math.max(0, Math.round(Date.now() / 1000 - tsSeconds))
    if (secs < 60) return `${secs}s ago`
    const mins = Math.floor(secs / 60)
    if (mins < 60) return `${mins}m ago`
    return `${Math.floor(mins / 60)}h ${mins % 60}m ago`
  }

  const runtime = status?.runtime || {}

  const backend = status?.backend || {}
  const model = status?.model || {}
  const online = probe.online === true
  const tone = probe.online == null ? 'stable' : online ? latencyTone(probe.latencyMs) : 'critical'

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="muted small">Service status · live probes from this browser</p>
          <h1>Admin — uptime &amp; hosting</h1>
          <p className="lede">
            Real-time backend reachability, latency, and deployment detail. Research prototype — values below are
            measured live, not synthetic.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-quiet btn-sm" onClick={check}>
            Probe now
          </button>
        </div>
      </header>

      <div className={`notice ${online === false ? 'notice-error' : ''}`} role="status">
        {probe.checking || probe.online == null
          ? 'Probing the backend…'
          : online
            ? `Backend reachable — ${probe.latencyMs} ms round trip at ${formatTime(probe.checkedAt)}.`
            : `Backend unreachable — last probe ${formatTime(probe.checkedAt)}: ${probe.error}. Showing last known detail.`}
      </div>

      <div className="admin-grid">
        <section className={`monitor tone-${tone}`} aria-label="Backend status">
          <div className="monitor-head">
            <strong>Backend · {online == null ? 'Probing' : online ? 'Up' : 'Down'}</strong>
            <span className={`status status-${tone} monitor-risk num`}>
              {probe.latencyMs == null ? '—' : `${probe.latencyMs} ms`}
            </span>
          </div>
          <dl className="admin-facts">
            <div><dt>Health</dt><dd className="num">{health?.status || status?.status || (online ? 'ok' : 'down')}</dd></div>
            <div><dt>Uptime</dt><dd className="num">{status?.uptime_human || health?.uptime_human || '—'}</dd></div>
            <div><dt>Uptime (s)</dt><dd className="num">{status?.uptime_seconds ?? health?.uptime_seconds ?? '—'}</dd></div>
            <div><dt>Session probes</dt><dd className="num">{stats.checks} checks · {stats.outages} outages</dd></div>
            <div><dt>Sampled uptime</dt><dd className="num">{stats.uptimePct == null ? '—' : `${stats.uptimePct}%`}</dd></div>
            <div><dt>Model loaded</dt><dd className="num">{String(model.model_loaded ?? health?.model_loaded ?? '—')}</dd></div>
          </dl>
        </section>

        <section className="monitor" aria-label="Throughput">
          <div className="monitor-head">
            <strong>Throughput · ingests/min</strong>
            <span className="monitor-risk num">
              {runtime.ingest_per_min_1m != null
                ? `${runtime.ingest_per_min_1m}/min`
                : throughput.latest == null ? '—' : `${throughput.latest.toFixed(1)}/min`}
            </span>
          </div>
          <div className="admin-latency">
            {throughputSpark ? (
              <svg viewBox="0 0 560 120" role="img" aria-label="Ingest rate history chart">
                <path d={throughputSpark} className="admin-spark" />
              </svg>
            ) : (
              <p className="muted small">Collecting samples — leave this page open while probes run every 5 s.</p>
            )}
            <dl className="admin-facts admin-facts-row">
              <div><dt>Now (1-min)</dt><dd className="num">{runtime.ingest_per_min_1m ?? '—'}</dd></div>
              <div><dt>5-min avg</dt><dd className="num">{runtime.ingest_per_min_5m ?? '—'}</dd></div>
              <div><dt>Session avg</dt><dd className="num">{throughput.avg == null ? '—' : throughput.avg.toFixed(1)}</dd></div>
              <div><dt>Total</dt><dd className="num">{runtime.ingest_count ?? '—'}</dd></div>
              <div><dt>Last ingest</dt><dd className="num">{formatAgo(runtime.last_ingest_time)}</dd></div>
            </dl>
          </div>
        </section>

        <section className="monitor" aria-label="Latency">
          <div className="monitor-head">
            <strong>Latency · last {history.length} probes</strong>
            <span className="monitor-risk num">{stats.avg == null ? '—' : `avg ${stats.avg} ms`}</span>
          </div>
          <div className="admin-latency">
            {spark ? (
              <svg viewBox="0 0 560 120" role="img" aria-label="Latency history chart">
                <path d={spark} className="admin-spark" />
              </svg>
            ) : (
              <p className="muted small">Collecting samples — leave this page open while probes run every 5 s.</p>
            )}
            <dl className="admin-facts admin-facts-row">
              <div><dt>Current</dt><dd className="num">{probe.latencyMs == null ? '—' : `${probe.latencyMs} ms`}</dd></div>
              <div><dt>Avg</dt><dd className="num">{stats.avg == null ? '—' : `${stats.avg} ms`}</dd></div>
              <div><dt>Min</dt><dd className="num">{stats.min == null ? '—' : `${stats.min} ms`}</dd></div>
              <div><dt>Max</dt><dd className="num">{stats.max == null ? '—' : `${stats.max} ms`}</dd></div>
            </dl>
          </div>
        </section>

        <section className="monitor" aria-label="Frontend hosting">
          <div className="monitor-head">
            <strong>Frontend · Vercel</strong>
            <span className="monitor-risk num">v{APP_VERSION}</span>
          </div>
          <dl className="admin-facts">
            <div><dt>Host</dt><dd className="num">{window.location.hostname}</dd></div>
            <div><dt>URL</dt><dd className="admin-wrap">{window.location.href}</dd></div>
            <div><dt>Website version</dt><dd className="num">v{APP_VERSION}</dd></div>
            <div><dt>Backend reports</dt><dd className="num">{version?.website_version ? `v${version.website_version}` : backend.website_version ? `v${backend.website_version}` : '—'}</dd></div>
            <div><dt>Build mode</dt><dd className="num">{import.meta.env.MODE}</dd></div>
            <div><dt>API target</dt><dd className="admin-wrap">{API_URL}</dd></div>
          </dl>
        </section>

        <section className="monitor" aria-label="Backend hosting">
          <div className="monitor-head">
            <strong>Backend · Render</strong>
            <span className="monitor-risk num">{backend.render_service || 'render'}</span>
          </div>
          <dl className="admin-facts">
            <div><dt>Service</dt><dd className="num">{backend.service || version?.service || 'syncura-backend'}</dd></div>
            <div><dt>Render service</dt><dd className="num">{backend.render_service || version?.render_service || '—'}</dd></div>
            <div><dt>External URL</dt><dd className="admin-wrap">{backend.render_external_url || version?.render_external_url || API_URL}</dd></div>
            <div><dt>Git commit</dt><dd className="num">{backend.git_commit || version?.git_commit || 'unknown'}</dd></div>
            <div><dt>Python</dt><dd className="num">{backend.python_version || version?.python_version || '—'}</dd></div>
            <div><dt>Platform</dt><dd className="admin-wrap">{backend.platform || '—'}</dd></div>
            <div><dt>Model</dt><dd className="num">{model.model_id || version?.model_id || '—'}</dd></div>
            <div><dt>Ensemble</dt><dd className="num">{model.ensemble_members ?? health?.ensemble_members ?? '—'} members</dd></div>
            <div><dt>Ingests served</dt><dd className="num">{status?.runtime?.ingest_count ?? '—'}</dd></div>
          </dl>
        </section>
      </div>

      <section aria-label="Uptime and downtime events">
        <h2>Uptime / downtime log</h2>
        <p className="muted small">Transitions observed by this browser, newest first. Persisted locally for the ops view.</p>
        {events.length === 0 ? (
          <p className="muted">No transitions recorded yet — probes run every 5 s while this page is open.</p>
        ) : (
          <div className="table-wrap">
            <table className="feed-table">
              <thead>
                <tr><th scope="col">Time</th><th scope="col">State</th><th scope="col">Detail</th></tr>
              </thead>
              <tbody>
                {events.map((e, i) => (
                  <tr key={`${e.t}-${i}`}>
                    <td className="num">{formatTime(e.t)}</td>
                    <td><span className={`status status-${e.online ? 'stable' : 'critical'}`}>{e.online ? 'Up' : 'Down'}</span></td>
                    <td>{e.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
