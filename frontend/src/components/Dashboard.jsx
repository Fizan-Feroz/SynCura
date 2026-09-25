import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useSimulation } from '../simulationContext'
import { API_URL } from '../api'
import { gsap, Flip, REDUCED } from '../motion/gsap'
import AppShell from './AppShell'
import { riskLabel, riskTone, seriesPath } from './trace'

const RERANK_MS = 5000

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function scoreContributions(vitals) {
  return [
    { name: 'Heart rate', value: (vitals.HR - 85) * 0.24 },
    { name: 'SpO2', value: (92 - vitals.SpO2) * 1.7 },
    { name: 'Respiratory rate', value: (vitals.Resp - 18) * 0.6 },
    { name: 'Temperature', value: (vitals.Temp - 37) * 4.5 },
  ].map((m) => ({ ...m, points: Number((m.value * 0.05).toFixed(2)) }))
}

function buildAlerts(patients) {
  const alerts = []
  patients.forEach((p) => {
    if (p.risk >= 90) {
      alerts.push({ key: `${p.patient_id}-risk`, level: 'critical', bed: p.bed, text: `Risk ${p.risk}%. Review at the bedside now.` })
      return
    }
    if (p.vitals.SpO2 <= 88) alerts.push({ key: `${p.patient_id}-spo2`, level: 'warning', bed: p.bed, text: `SpO2 low at ${p.vitals.SpO2}%.` })
    if (p.vitals.Resp >= 30) alerts.push({ key: `${p.patient_id}-rr`, level: 'warning', bed: p.bed, text: `Respiratory rate high at ${p.vitals.Resp}/min.` })
    if (p.vitals.Temp >= 39) alerts.push({ key: `${p.patient_id}-temp`, level: 'info', bed: p.bed, text: `Temperature ${p.vitals.Temp} °C, possible infection.` })
  })
  return alerts.slice(0, 6)
}

function calculateNews2({ vitals: { HR, Resp, Temp, SpO2 } }) {
  let score = 0
  if (Resp <= 8 || Resp >= 25) score += 3
  else if (Resp >= 21) score += 2
  else if (Resp >= 9 && Resp <= 11) score += 1
  if (SpO2 <= 91) score += 3
  else if (SpO2 <= 93) score += 2
  else if (SpO2 <= 95) score += 1
  if (Temp <= 35) score += 3
  else if (Temp >= 39.1) score += 2
  else if (Temp >= 38.1) score += 1
  if (HR <= 40 || HR >= 131) score += 3
  else if (HR >= 111) score += 2
  else if (HR >= 91 || HR <= 50) score += 1
  return score
}

function hasDeteriorationEvent(p) {
  return p.vitals.SpO2 <= 90 || p.vitals.Resp >= 30 || p.vitals.Temp >= 39.2 || p.risk >= 88
}

function classificationStats(rows) {
  const t = rows.reduce(
    (acc, r) => {
      if (r.predicted && r.actual) acc.tp += 1
      else if (r.predicted) acc.fp += 1
      else if (r.actual) acc.fn += 1
      else acc.tn += 1
      return acc
    },
    { tp: 0, fp: 0, tn: 0, fn: 0 }
  )
  const pct = (a, b) => (b ? Number(((a / b) * 100).toFixed(1)) : 0)
  return { ...t, sensitivity: pct(t.tp, t.tp + t.fn), specificity: pct(t.tn, t.tn + t.fp), precision: pct(t.tp, t.tp + t.fp) }
}

function RiskRing({ value, size = 56 }) {
  const r = 24
  const c = 2 * Math.PI * r
  return (
    <span className={`risk-ring tone-${riskTone(value)}`} style={{ width: size, height: size }}>
      <svg viewBox="0 0 56 56" aria-hidden="true">
        <circle className="risk-ring-track" cx="28" cy="28" r={r} />
        <circle
          className="risk-ring-value"
          cx="28"
          cy="28"
          r={r}
          strokeDasharray={c}
          strokeDashoffset={c * (1 - clamp(value, 0, 100) / 100)}
        />
      </svg>
      <span className="risk-ring-num num">{value}</span>
    </span>
  )
}

function Readout({ label, value, unit, color, big = false }) {
  return (
    <span className={`readout ${big ? 'readout-big' : ''}`} style={{ '--c': color }}>
      <span className="readout-label">{label}</span>
      <span className="readout-value num">
        {value}
        {unit && <small>{unit}</small>}
      </span>
    </span>
  )
}

function BedTile({ patient, selected, onSelect, threshold }) {
  const tone = riskTone(patient.risk)
  const trend = seriesPath(patient.waveform, 200, 44, 3, [0, 100])
  const over = patient.risk >= threshold
  return (
    <li className="bed" data-flip-id={patient.patient_id}>
      <button
        type="button"
        className={`bed-tile tone-${tone} ${selected ? 'is-selected' : ''} ${over ? 'is-over' : ''}`}
        onClick={() => onSelect(patient.patient_id)}
        aria-pressed={selected}
        aria-label={`${patient.bed}, patient ${patient.patient_id}, risk ${patient.risk} percent, ${riskLabel(patient.risk)}`}
      >
        <span className="bed-head">
          <span className="bed-name">{patient.bed}</span>
          <span className={`status status-${tone}`}>{riskLabel(patient.risk)}</span>
        </span>
        <span className="bed-body">
          <span className="bed-readouts">
            <Readout label="HR" value={patient.vitals.HR} color="var(--hr)" />
            <Readout label="SpO2" value={patient.vitals.SpO2} color="var(--spo2)" />
            <Readout label="RR" value={patient.vitals.Resp} color="var(--rr)" />
            <Readout label="T" value={patient.vitals.Temp.toFixed(1)} color="var(--temp)" />
          </span>
          <RiskRing value={patient.risk} />
        </span>
        <svg className="bed-trend" viewBox="0 0 200 44" preserveAspectRatio="none" aria-hidden="true">
          <path d={trend} vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="bed-foot">
          <span>{patient.lead}</span>
          <span className="num">{patient.trend}</span>
        </span>
      </button>
    </li>
  )
}

export default function Dashboard({ theme, onToggleTheme }) {
  const {
    activeScenario,
    scenarioEntries,
    patientQueue,
    lastUpdated,
    isPaused,
    setActiveScenario,
    toggleSimulation,
    resetSimulation,
  } = useSimulation()
  const [selectedId, setSelectedId] = useState(patientQueue[0]?.patient_id)
  const [threshold, setThreshold] = useState(75)
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState(false)

  useEffect(() => {
    fetch(`${API_URL}/metrics`)
      .then((r) => r.json())
      .then((d) => (d.error ? setMetricsError(true) : setMetrics(d)))
      .catch(() => setMetricsError(true))
  }, [])

  // Ranking: re-sorted every few seconds rather than on every tick, the way a
  // central station re-orders beds, so a move is meaningful and visible.
  const queueRef = useRef(patientQueue)
  queueRef.current = patientQueue
  const rank = () => [...queueRef.current].sort((a, b) => b.risk - a.risk).map((p) => p.patient_id)
  const [order, setOrder] = useState(rank)
  useEffect(() => {
    setOrder(rank())
    if (isPaused) return undefined
    const id = window.setInterval(() => setOrder(rank()), RERANK_MS)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPaused])
  useEffect(() => {
    // Reset or new patients: re-rank immediately.
    const ids = new Set(patientQueue.map((p) => p.patient_id))
    if (order.length !== ids.size || order.some((id) => !ids.has(id))) setOrder(rank())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientQueue])

  const byId = useMemo(() => new Map(patientQueue.map((p) => [p.patient_id, p])), [patientQueue])
  const ranked = order.map((id) => byId.get(id)).filter(Boolean)

  // FLIP: record tile positions before React commits a new order...
  const gridRef = useRef(null)
  const flipState = useRef(null)
  const orderKey = order.join(',')
  const lastKey = useRef(orderKey)
  if (orderKey !== lastKey.current && gridRef.current) {
    flipState.current = Flip.getState(gridRef.current.querySelectorAll('.bed'))
  }
  // ...then animate from those positions to the new ones.
  useLayoutEffect(() => {
    lastKey.current = orderKey
    const state = flipState.current
    flipState.current = null
    if (!state || window.matchMedia(REDUCED).matches) return undefined
    const anim = Flip.from(state, {
      duration: 0.75,
      ease: 'power3.inOut',
      stagger: 0.02,
      onEnter: (els) => gsap.fromTo(els, { opacity: 0 }, { opacity: 1, duration: 0.4 }),
    })
    return () => anim.kill()
  }, [orderKey])

  const selected = byId.get(selectedId) || ranked[0]
  const needReview = patientQueue.filter((p) => p.risk >= threshold).length
  const alerts = useMemo(() => buildAlerts(patientQueue), [patientQueue])
  const modelPerf = useMemo(
    () => classificationStats(patientQueue.map((p) => ({ actual: hasDeteriorationEvent(p), predicted: p.risk >= threshold }))),
    [patientQueue, threshold]
  )
  const news2Perf = useMemo(
    () => classificationStats(patientQueue.map((p) => ({ actual: hasDeteriorationEvent(p), predicted: calculateNews2(p) >= 7 }))),
    [patientQueue]
  )
  const leadTime = useMemo(() => {
    const lead = patientQueue.map((p) => clamp((100 - p.risk) / 12, 0.5, 6))
    return (lead.reduce((s, v) => s + v, 0) / Math.max(1, lead.length)).toFixed(1)
  }, [patientQueue])

  const fmt = (v, pct = true) => (v == null ? 'n/a' : pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(3))

  return (
    <AppShell
      theme={theme}
      onToggleTheme={onToggleTheme}
      wide
      status={{ live: !isPaused, label: isPaused ? 'Stream paused' : 'Streaming to backend' }}
    >
      <div className="station">
        <header className="station-head">
          <div>
            <h1>Central station</h1>
            <p className="muted small">
              Synthetic patients. Scores on this screen are simulated in your browser; every reading is also sent
              to the backend model.
            </p>
          </div>
          <div className="station-controls">
            <div className="segmented" role="group" aria-label="Scenario">
              {scenarioEntries.map(([key, s]) => (
                <button key={key} type="button" aria-pressed={activeScenario === key} onClick={() => setActiveScenario(key)}>
                  {s.label}
                </button>
              ))}
            </div>
            <div className="station-actions">
              <button type="button" className="btn btn-primary btn-sm" onClick={toggleSimulation}>
                {isPaused ? 'Start stream' : 'Pause stream'}
              </button>
              <button type="button" className="btn btn-quiet btn-sm" onClick={resetSimulation}>
                Reset beds
              </button>
            </div>
          </div>
        </header>

        <dl className="station-summary" aria-label="Unit summary">
          <div className={needReview ? 'is-alarm' : ''}>
            <dt>Need review</dt>
            <dd className="num">{needReview}</dd>
          </div>
          <div>
            <dt>Beds</dt>
            <dd className="num">{patientQueue.length}</dd>
          </div>
          <div>
            <dt>Lead time, simulated</dt>
            <dd className="num">{leadTime}<small>h</small></dd>
          </div>
          <div>
            <dt>False alarms now</dt>
            <dd className="num">{modelPerf.fp}</dd>
          </div>
          <div>
            <dt>Last update</dt>
            <dd className="num">{lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</dd>
          </div>
        </dl>

        {isPaused && (
          <p className="station-hint">
            The stream is paused. Start it to watch the beds change and re-rank every five seconds.
          </p>
        )}

        <div className="station-body">
          <section aria-labelledby="beds-title">
            <h2 id="beds-title" className="sr-only">Beds ranked by risk</h2>
            <ol className="beds" ref={gridRef}>
              {ranked.map((p) => (
                <BedTile key={p.patient_id} patient={p} selected={p.patient_id === selected?.patient_id} onSelect={setSelectedId} threshold={threshold} />
              ))}
            </ol>
          </section>

          <aside className="station-rail">
            <section className="rail-block" aria-labelledby="alerts-title" aria-live="polite">
              <div className="rail-head">
                <h2 id="alerts-title">Alerts</h2>
                <span className="num muted">{alerts.length}</span>
              </div>
              {alerts.length === 0 ? (
                <p className="muted small">No alarm limits crossed. Monitoring continues.</p>
              ) : (
                <ul className="alerts">
                  {alerts.map((a) => (
                    <li key={a.key} className={`alert alert-${a.level}`}>
                      <span className="alert-bed">{a.bed}</span>
                      <span>{a.text}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {selected && (
              <section className={`rail-block detail tone-${riskTone(selected.risk)}`} aria-labelledby="detail-title">
                <div className="rail-head">
                  <h2 id="detail-title">
                    {selected.bed}
                    <span className="muted"> patient {selected.patient_id}</span>
                  </h2>
                  <RiskRing value={selected.risk} size={64} />
                </div>
                <div className="detail-readouts">
                  <Readout big label="HR" value={selected.vitals.HR} unit="bpm" color="var(--hr)" />
                  <Readout big label="SpO2" value={selected.vitals.SpO2} unit="%" color="var(--spo2)" />
                  <Readout big label="RR" value={selected.vitals.Resp} unit="/min" color="var(--rr)" />
                  <Readout big label="Temp" value={selected.vitals.Temp.toFixed(1)} unit="°C" color="var(--temp)" />
                </div>
                <svg className="detail-trend" viewBox="0 0 320 90" preserveAspectRatio="none" role="img" aria-label={`Risk trend for ${selected.bed}`}>
                  <line x1="0" x2="320" y1={90 - (threshold / 100) * 90} y2={90 - (threshold / 100) * 90} className="detail-threshold" vectorEffect="non-scaling-stroke" />
                  <path d={seriesPath(selected.waveform, 320, 90, 4, [0, 100])} vectorEffect="non-scaling-stroke" />
                </svg>
                <p className="small muted">Dashed line: your alert threshold ({threshold}). NEWS2 for this bed: <strong className="num">{calculateNews2(selected)}</strong>.</p>
                <h3 className="detail-sub">What pushes this score</h3>
                <ul className="drivers">
                  {scoreContributions(selected.vitals).map((m) => (
                    <li key={m.name}>
                      <span>{m.name}</span>
                      <span className={`driver-bar ${m.points >= 0 ? 'up' : 'down'}`}>
                        <span style={{ '--w': Math.min(1, Math.abs(m.value) / 10) }} />
                      </span>
                      <span className="num">{m.points >= 0 ? '+' : ''}{m.points}</span>
                    </li>
                  ))}
                </ul>
                <p className="small muted">A simplified view of the simulation, not the model’s attention or SHAP values.</p>
              </section>
            )}
          </aside>
        </div>

        <section className="station-lower" aria-label="Alert tuning and model">
          <div className="lower-block">
            <h2>Alert threshold</h2>
            <label className="threshold" htmlFor="threshold">
              <span>Alert when risk is at least <strong className="num">{threshold}</strong></span>
              <input id="threshold" type="range" min="50" max="95" step="1" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            </label>
            <dl className="stat-row">
              <div><dt>Sensitivity</dt><dd className="num">{modelPerf.sensitivity}%</dd></div>
              <div><dt>Specificity</dt><dd className="num">{modelPerf.specificity}%</dd></div>
              <div><dt>Precision</dt><dd className="num">{modelPerf.precision}%</dd></div>
            </dl>
          </div>

          <div className="lower-block">
            <h2>Against NEWS2</h2>
            <table className="compare">
              <thead>
                <tr><th scope="col"><span className="sr-only">Method</span></th><th scope="col">Sensitivity</th><th scope="col">Specificity</th></tr>
              </thead>
              <tbody>
                <tr><th scope="row">SynCura at {threshold}</th><td className="num">{modelPerf.sensitivity}%</td><td className="num">{modelPerf.specificity}%</td></tr>
                <tr><th scope="row">NEWS2 at 7 or more</th><td className="num">{news2Perf.sensitivity}%</td><td className="num">{news2Perf.specificity}%</td></tr>
              </tbody>
            </table>
            <p className="small muted">Computed on the simulated beds, so it shows the idea, not real performance.</p>
          </div>

          <div className="lower-block">
            <h2>Deployed model</h2>
            {metricsError ? (
              <p className="small muted">Start the backend to load the model’s offline metrics.</p>
            ) : (
              <dl className="stat-row">
                <div><dt>AUC</dt><dd className="num">{metrics ? fmt(metrics.auc, false) : '…'}</dd></div>
                <div><dt>Recall</dt><dd className="num">{metrics ? fmt(metrics.recall) : '…'}</dd></div>
                <div><dt>Precision</dt><dd className="num">{metrics ? fmt(metrics.precision) : '…'}</dd></div>
              </dl>
            )}
            <p className="small muted">Three-model ensemble, offline PhysioNet evaluation. <Link to="/architecture">Read the caveats</Link>.</p>
          </div>
        </section>
      </div>
    </AppShell>
  )
}
