import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { API_URL } from './api'

export const SCENARIOS = {
  baseline: {
    label: 'Baseline Mix',
    lead: 'Mixed instability',
    profile: { hr: 0, spo2: 0, resp: 0, temp: 0, riskDrift: 0, volatility: 1.5 },
  },
  respiratory: {
    label: 'Respiratory Decline',
    lead: 'Respiratory decline',
    profile: { hr: 4, spo2: -3, resp: 5, temp: 0.3, riskDrift: 4, volatility: 2.5 },
  },
  septic: {
    label: 'Septic Shock',
    lead: 'Sepsis escalation',
    profile: { hr: 8, spo2: -2, resp: 4, temp: 0.8, riskDrift: 6, volatility: 3 },
  },
  cardiac: {
    label: 'Cardiac Stress',
    lead: 'Arrhythmic stress',
    profile: { hr: 11, spo2: -1, resp: 2, temp: 0.2, riskDrift: 5, volatility: 4 },
  },
  recovery: {
    label: 'Recovery Trend',
    lead: 'Clinical recovery',
    profile: { hr: -5, spo2: 2, resp: -3, temp: -0.4, riskDrift: -5, volatility: 1.2 },
  },
}

const SimulationContext = createContext(null)

const BACKEND_CHECK_MS = 15000
const BACKEND_TIMEOUT_MS = 30000
const BACKEND_VERSION_TIMEOUT_MS = 8000

function statusForRisk(risk) {
  if (risk >= 85) return 'Critical'
  if (risk >= 70) return 'High'
  if (risk >= 45) return 'Watch'
  return 'Stable'
}

function pickLatestNumber(rows, key) {
  for (const row of rows) {
    const value = Number(row?.[key])
    if (Number.isFinite(value)) return value
  }
  return NaN
}

// Map a backend /patient/{id} payload (replayed PhysioNet rows, device
// ingests, or anything else in the backend store) onto the bed shape the
// views render. Missing vitals stay NaN so thresholds never misfire.
function mapLivePatient(detail) {
  const rows = Array.isArray(detail?.vitals) ? detail.vitals : []
  const id = String(detail?.patient_id ?? 'unknown')
  const scores = rows.map((row) => Number(row?.risk_score)).filter(Number.isFinite).reverse()
  const waveform = (scores.length ? scores : [Number(detail?.risk) || 0]).slice(-12).map((v) => Math.round(v))
  const risk = waveform.length ? waveform[waveform.length - 1] : 0
  const delta = waveform.length > 1 ? waveform[waveform.length - 1] - waveform[0] : 0
  return {
    patient_id: id,
    bed: `BED-${id.slice(-3)}`,
    status: statusForRisk(risk),
    risk,
    trend: `${delta >= 0 ? '+' : ''}${delta}`,
    lead: 'Backend ingest',
    trajectory: 'replay',
    waveform,
    vitals: {
      HR: pickLatestNumber(rows, 'hr'),
      SpO2: pickLatestNumber(rows, 'spo2'),
      Resp: pickLatestNumber(rows, 'rr'),
      Temp: pickLatestNumber(rows, 'temp'),
    },
  }
}

export function SimulationProvider({ children }) {
  const [patientQueue, setPatientQueue] = useState([])
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendError, setBackendError] = useState(null)
  const [backendVersion, setBackendVersion] = useState(null)
  // Data source: 'live' mirrors the shared backend scenario engine so every
  // browser sees identical beds; 'replay' shows whatever the backend has
  // actually ingested (PhysioNet replay or device posts). There is no local
  // simulation anymore — one shared state for everyone.
  const [source, setSource] = useState('live')
  const [liveMeta, setLiveMeta] = useState({ scenario: 'baseline', scenarioLabel: 'Baseline Mix', tick: 0, paused: true })
  const [liveError, setLiveError] = useState(null)
  const checkSeqRef = React.useRef(0)
  const healthControllerRef = React.useRef(null)
  const versionControllerRef = React.useRef(null)

  const checkBackend = useCallback(async ({ indicate = false } = {}) => {
    // Serialize checks: abort the previous request and ignore any response
    // that does not belong to the latest check. This keeps a slow retry from
    // being overwritten by an earlier in-flight poll.
    checkSeqRef.current += 1
    const seq = checkSeqRef.current
    const isCurrent = () => checkSeqRef.current === seq
    healthControllerRef.current?.abort()
    versionControllerRef.current?.abort()

    const healthController = new AbortController()
    healthControllerRef.current = healthController
    const timeoutId = window.setTimeout(() => healthController.abort(), BACKEND_TIMEOUT_MS)
    if (indicate && isCurrent()) {
      setBackendStatus('checking')
      setBackendError(null)
    }
    try {
      const response = await fetch(`${API_URL}/health`, { signal: healthController.signal })
      if (!response.ok) throw new Error(`GET /health -> ${response.status}`)
      await response.json().catch(() => ({}))
      if (!isCurrent()) return false

      let deployedWebsiteVersion = null
      const versionController = new AbortController()
      versionControllerRef.current = versionController
      const versionTimeoutId = window.setTimeout(
        () => versionController.abort(),
        BACKEND_VERSION_TIMEOUT_MS
      )
      try {
        const versionResponse = await fetch(`${API_URL}/version`, { signal: versionController.signal })
        if (versionResponse.ok) {
          const versionData = await versionResponse.json().catch(() => ({}))
          if (typeof versionData?.website_version === 'string') {
            deployedWebsiteVersion = versionData.website_version
          }
        }
      } catch {
        // Health is sufficient for online status; version tracking is best-effort.
      } finally {
        window.clearTimeout(versionTimeoutId)
        if (versionControllerRef.current === versionController) versionControllerRef.current = null
      }
      if (!isCurrent()) return false
      setBackendStatus('online')
      setBackendError(null)
      setBackendVersion(deployedWebsiteVersion)
      return true
    } catch (error) {
      if (!isCurrent()) return false
      setBackendStatus('offline')
      setBackendVersion(null)
      setBackendError(
        error?.name === 'AbortError'
          ? `The backend did not respond within ${BACKEND_TIMEOUT_MS / 1000} seconds.`
          : error?.message || 'The backend could not be reached.'
      )
      return false
    } finally {
      window.clearTimeout(timeoutId)
      if (healthControllerRef.current === healthController) healthControllerRef.current = null
    }
  }, [])

  const retryBackend = useCallback(() => checkBackend({ indicate: true }), [checkBackend])

  // Views only render when the backend is reachable; the gate lives in the
  // route components via BackendStatusPanel.
  useEffect(() => {
    let cancelled = false
    let intervalId
    const check = async () => {
      if (document.visibilityState === 'hidden') return
      // Never start a scheduled poll while another check is running. Manual
      // retries abort the in-flight check and take priority instead.
      if (healthControllerRef.current) return
      if (!cancelled) await checkBackend()
    }
    check()
    intervalId = window.setInterval(check, BACKEND_CHECK_MS)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [checkBackend])

  // Backend live mode: mirror the shared scenario engine.
  const reloadLive = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/simulation/state`)
      if (!res.ok) throw new Error(`GET /simulation/state -> ${res.status}`)
      const data = await res.json()
      const beds = Array.isArray(data.beds) ? data.beds : []
      setPatientQueue(beds)
      setLiveMeta({
        scenario: data.scenario || 'baseline',
        scenarioLabel: data.scenario_label || data.scenario || 'Baseline',
        tick: data.tick ?? 0,
        paused: data.paused ?? true,
        simulated: data.simulated ?? data.scenario !== 'baseline',
      })
      setLiveError(beds.length ? null : 'empty')
      setLastUpdated(new Date())
      return true
    } catch (err) {
      setLiveError(err?.message || 'Live state unreachable.')
      return false
    }
  }, [])

  useEffect(() => {
    if (source !== 'live' || backendStatus !== 'online') return undefined
    reloadLive()
    const intervalId = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') reloadLive()
    }, 5000)
    return () => window.clearInterval(intervalId)
  }, [source, backendStatus, reloadLive])

  // Replay source: read back what the backend actually ingested. Works with
  // PhysioNet replay (backend/replay.py), device posts, or simulation ingests.
  const reloadReplay = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/patients`)
      if (!res.ok) throw new Error(`GET /patients -> ${res.status}`)
      const data = await res.json()
      const list = Array.isArray(data.patients) ? data.patients.slice(0, 12) : []
      const details = await Promise.all(
        list.map(async (entry) => {
          try {
            const r = await fetch(`${API_URL}/patient/${encodeURIComponent(entry.patient_id)}`)
            if (!r.ok) return null
            return r.json()
          } catch {
            return null
          }
        })
      )
      const beds = details.filter(Boolean).map(mapLivePatient)
      setPatientQueue(beds)
      setLiveError(beds.length ? null : 'empty')
      setLastUpdated(new Date())
      return true
    } catch (err) {
      setLiveError(err?.message || 'Backend patients unreachable.')
      return false
    }
  }, [])

  useEffect(() => {
    if (source !== 'replay' || backendStatus !== 'online') return undefined
    reloadReplay()
    const intervalId = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') reloadReplay()
    }, 5000)
    return () => window.clearInterval(intervalId)
  }, [source, backendStatus, reloadReplay])

  const controlLive = useCallback(async (action, scenario) => {
    try {
      const res = await fetch(`${API_URL}/simulation/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenario ? { action, scenario } : { action }),
      })
      if (!res.ok) throw new Error(`POST /simulation/control -> ${res.status}`)
      const data = await res.json()
      setPatientQueue(Array.isArray(data.beds) ? data.beds : [])
      setLiveMeta({
        scenario: data.scenario || 'baseline',
        scenarioLabel: data.scenario_label || data.scenario || 'Baseline',
        tick: data.tick ?? 0,
        paused: data.paused ?? true,
        simulated: data.simulated ?? data.scenario !== 'baseline',
      })
      setLiveError(null)
      setLastUpdated(new Date())
    } catch (err) {
      setLiveError(err?.message || 'Live control unreachable.')
    }
  }, [])

  const setDataSource = useCallback((next) => {
    if (next !== 'live' && next !== 'replay') return
    setSource(next)
    setLiveError(null)
    setLastUpdated(new Date())
  }, [])

  const value = useMemo(
    () => ({
      activeScenario: liveMeta.scenario,
      activeScenarioLabel: liveMeta.scenarioLabel,
      scenarioEntries: Object.entries(SCENARIOS),
      patientQueue,
      lastUpdated,
      isPaused: liveMeta.paused,
      backendOnline: backendStatus === 'online',
      backendChecking: backendStatus === 'checking',
      backendError,
      backendVersion,
      retryBackend,
      source,
      setDataSource,
      liveError,
      liveMeta,
      reloadLive,
      reloadReplay,
      setActiveScenario: (key) => controlLive('set_scenario', key),
      toggleSimulation: () => controlLive(liveMeta.paused ? 'start' : 'pause'),
      resetSimulation: () => controlLive('reset'),
    }),
    [backendError, backendStatus, backendVersion, controlLive, liveError, liveMeta, patientQueue, lastUpdated, reloadLive, reloadReplay, retryBackend, source]
  )

  return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>
}

export function BackendStatusPanel({ title, backendChecking, backendError, retryBackend }) {
  if (backendChecking) {
    return (
      <div className="page">
        <div className="skeleton" role="status">
          <span className="sr-only">Checking the backend connection…</span>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{title} unavailable</h1>
          <p className="muted">Live views stop while the backend is unreachable.</p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-primary btn-sm" onClick={retryBackend}>
            Retry connection
          </button>
        </div>
      </header>
      <p className="notice notice-error" role="alert">
        Backend unavailable. Live risk scores, model metrics, training jobs, and alert delivery need the
        backend, so this view is stopped instead of showing stale beds.
      </p>
      {backendError && <p className="muted small">{backendError}</p>}
    </div>
  )
}

export function useSimulation() {
  const context = useContext(SimulationContext)
  if (!context) {
    throw new Error('useSimulation must be used within SimulationProvider')
  }
  return context
}
