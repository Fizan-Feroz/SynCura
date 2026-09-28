import React, { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { apiGet } from '../api'
import { useSimulation } from '../simulationContext'
import AppShell from './AppShell'
import { riskLabel, riskTone, seriesPath } from './trace'
import { buildAlerts, calculateNews2, scoreContributions } from './alerts'

const CHANNELS = [
  { key: 'HR', label: 'Heart rate', short: 'HR', unit: 'bpm', color: 'var(--hr)' },
  { key: 'SpO2', label: 'SpO2', short: 'SpO2', unit: '%', color: 'var(--spo2)' },
  { key: 'Resp', label: 'Respiration', short: 'RR', unit: '/min', color: 'var(--rr)' },
  { key: 'Temp', label: 'Temperature', short: 'Temp', unit: '°C', color: 'var(--temp)', digits: 1 },
]

function fmtVital(value, digits = 0) {
  if (!Number.isFinite(value)) return '—'
  return digits ? value.toFixed(digits) : value
}

export default function BedProfile({ theme, onToggleTheme }) {
  const { patientId } = useParams()
  const { patientQueue, backendOnline, source } = useSimulation()
  const bed = patientQueue.find((p) => p.patient_id === patientId)
  const [shap, setShap] = useState(null)

  useEffect(() => {
    let cancelled = false
    setShap(null)
    if (!backendOnline || !bed) return undefined
    apiGet(`/patient/${encodeURIComponent(bed.patient_id)}/explain`)
      .then((body) => {
        if (cancelled) return
        const imp = body?.feature_importance
        if (imp && typeof imp === 'object' && !imp.error) {
          const top = Object.entries(imp)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3)
          setShap(top)
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [backendOnline, bed?.patient_id]) // eslint-disable-line react-hooks/exhaustive-deps

  const shell = (children) => (
    <AppShell theme={theme} onToggleTheme={onToggleTheme} wide>
      {children}
    </AppShell>
  )

  if (!bed) {
    return shell(
      <div className="page">
        <p>
          <Link to="/dashboard">← Back to central station</Link>
        </p>
        <h1>Bed not found</h1>
        <p className="muted">
          No bed with id <span className="num">{patientId}</span> in the current{' '}
          {source === 'live' ? 'backend live' : 'replay'} view. It may live in the other data
          source — switch source on the dashboard.
        </p>
      </div>
    )
  }

  const tone = riskTone(bed.risk)
  const alert = buildAlerts([bed])[0] || null
  const history = Array.isArray(bed.history) ? bed.history : []
  const drivers = scoreContributions(bed.vitals)

  return shell(
    <div className="page">
      <p>
        <Link to="/dashboard">← Back to central station</Link>
      </p>
      <header className="page-head">
        <div>
          <h1>
            {bed.bed} <span className="muted small">patient {bed.patient_id}</span>
          </h1>
          <p className="muted">
            <span className={`status status-${tone}`}>{riskLabel(bed.risk)}</span>
            {' · '}
            {source === 'live' ? 'Shared backend simulation' : 'Backend ingest · retrospective'} · {bed.lead}
          </p>
        </div>
        <div className="page-actions">
          <span className="readout readout-big" style={{ '--c': `var(--${tone === 'stable' ? 'stable' : tone})` }}>
            <span className="readout-label">Risk</span>
            <span className="readout-value num">
              {bed.risk}
              <small>/100</small>
            </span>
          </span>
        </div>
      </header>

      <div className="monitors">
        {CHANNELS.map((c) => {
          const values = history.map((h) => h[c.key]).filter(Number.isFinite)
          return (
            <article key={c.key} className="monitor">
              <header className="monitor-head">
                <strong>{c.label}</strong>
                <span className="num channel-value" style={{ '--c': c.color }}>
                  {fmtVital(bed.vitals[c.key], c.digits || 0)}
                  <small>{c.unit}</small>
                </span>
              </header>
              {values.length > 1 ? (
                <svg viewBox="0 0 300 48" preserveAspectRatio="none" role="img" aria-label={`${c.label} history for ${bed.bed}`}>
                  <path d={seriesPath(values, 300, 48, 4)} fill="none" style={{ stroke: c.color }} strokeWidth="2" vectorEffect="non-scaling-stroke" />
                </svg>
              ) : (
                <p className="muted small monitor-empty">History builds as the stream runs.</p>
              )}
            </article>
          )
        })}
      </div>

      <section className="rail-block" aria-label="Risk trend" style={{ marginTop: 'var(--space-5)' }}>
        <div className="rail-head">
          <h2>Risk trend</h2>
          <span className="num muted">{bed.trend}</span>
        </div>
        <svg className="detail-trend" viewBox="0 0 320 90" preserveAspectRatio="none" role="img" aria-label={`Risk trend for ${bed.bed}`}>
          <path d={seriesPath(bed.waveform, 320, 90, 4, [0, 100])} vectorEffect="non-scaling-stroke" />
        </svg>
        <p className="small muted">
          NEWS2 for this bed: <strong className="num">{calculateNews2(bed)}</strong> · Trajectory:{' '}
          {bed.trajectory || 'stable'}
        </p>
      </section>

      {alert && (
        <section className="rail-block" aria-label="Active alert" style={{ marginTop: 'var(--space-5)' }}>
          <div className="rail-head">
            <h2>Active alert</h2>
          </div>
          <ul className="alerts">
            <li className={`alert alert-${alert.level}`}>
              <span className="alert-bed">{alert.bed}</span>
              <span className="alert-body">
                <strong className="alert-title">{alert.title}</strong>
                {alert.condition && <span className="alert-condition">{alert.condition}</span>}
                <span>{alert.signal}</span>
                <span className="alert-cause"><strong>Why:</strong> {alert.cause}</span>
                <span className="alert-action"><strong>Do:</strong> {alert.action}</span>
              </span>
            </li>
          </ul>
        </section>
      )}

      <section className="rail-block" aria-label="What pushes this score" style={{ marginTop: 'var(--space-5)' }}>
        <div className="rail-head">
          <h2>What pushes this score</h2>
        </div>
        <ul className="drivers">
          {drivers.map((m) => (
            <li key={m.name}>
              <span>{m.name}</span>
              <span className={`driver-bar ${m.points >= 0 ? 'up' : 'down'}`}>
                <span style={{ '--w': Number.isFinite(m.value) ? Math.min(1, Math.abs(m.value) / 10) : 0 }} />
              </span>
              <span className="num">{Number.isFinite(m.points) ? `${m.points >= 0 ? '+' : ''}${m.points}` : '—'}</span>
            </li>
          ))}
        </ul>
        {shap ? (
          <p className="small muted">
            Backend model agrees on: {shap.map(([name]) => name).join(', ')} (top SHAP features).
          </p>
        ) : (
          <p className="small muted">A simplified view of the bed state, not the model’s attention or SHAP values.</p>
        )}
      </section>

      <p className="muted small" style={{ marginTop: 'var(--space-5)' }}>
        {source === 'live'
          ? 'Shared backend simulation — demo, not for clinical use.'
          : 'Backend-ingested retrospective data — demo, not for clinical use.'}
      </p>
    </div>
  )
}
