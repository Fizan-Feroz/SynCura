import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import mimicDemoPatients from './mimicDemoPatients.json'
import { API_URL } from './api'

export const BASE_PATIENTS = [
  {
    patient_id: '132547',
    bed: 'ICU-04',
    status: 'High',
    risk: 76,
    trend: '+2',
    lead: 'Mixed instability',
    vitals: { HR: 112, SpO2: 91, Resp: 27, Temp: 38.0 },
    waveform: [56, 60, 58, 61, 64, 62, 66, 69, 68, 71, 74, 76],
  },
  {
    patient_id: '132611',
    bed: 'ICU-09',
    status: 'Watch',
    risk: 63,
    trend: '+1',
    lead: 'Hemodynamic watch',
    vitals: { HR: 96, SpO2: 94, Resp: 22, Temp: 37.6 },
    waveform: [44, 47, 45, 49, 52, 50, 53, 57, 58, 60, 61, 63],
  },
  {
    patient_id: '132590',
    bed: 'ICU-12',
    status: 'Watch',
    risk: 52,
    trend: '+0',
    lead: 'Early inflammatory signal',
    vitals: { HR: 90, SpO2: 95, Resp: 21, Temp: 37.5 },
    waveform: [36, 38, 39, 37, 41, 40, 43, 44, 45, 48, 49, 52],
  },
  {
    patient_id: '132539',
    bed: 'ICU-02',
    status: 'Stable',
    risk: 31,
    trend: '-1',
    lead: 'Baseline recovery',
    vitals: { HR: 75, SpO2: 97, Resp: 18, Temp: 36.7 },
    waveform: [38, 37, 35, 36, 33, 34, 32, 31, 30, 32, 30, 31],
  },
]

const DEFAULT_PATIENTS = Array.isArray(mimicDemoPatients) && mimicDemoPatients.length >= 12
  ? mimicDemoPatients.slice(0, 12)
  : BASE_PATIENTS

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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function randomCentered(scale) {
  return (Math.random() * 2 - 1) * scale
}

function randomPick(items) {
  return items[Math.floor(Math.random() * items.length)]
}

function assignTrajectory() {
  const roll = Math.random()
  if (roll < 0.27) return 'severe'
  if (roll < 0.52) return 'recovery'
  if (roll < 0.78) return 'stable'
  return 'volatile'
}

function trajectoryProfile(trajectory) {
  switch (trajectory) {
    case 'severe':
      return { hr: 7, spo2: -3, resp: 5, temp: 0.4, riskDrift: 7, volatility: 2.6, lead: 'Multi-organ deterioration' }
    case 'recovery':
      return { hr: -4, spo2: 2, resp: -3, temp: -0.3, riskDrift: -6, volatility: 1.4, lead: 'Clinical improvement' }
    case 'volatile':
      return { hr: 3, spo2: -1, resp: 2, temp: 0.1, riskDrift: 1, volatility: 4.2, lead: 'Unstable oscillations' }
    case 'stable':
    default:
      return { hr: 0, spo2: 0, resp: 0, temp: 0, riskDrift: 0, volatility: 1.2, lead: 'Stable monitoring' }
  }
}

function seedIcuEnvironment(patients) {
  const beds = patients.map((p) => p.bed)
  const takenBeds = new Set()

  return patients.map((patient, index) => {
    const trajectory = assignTrajectory()
    const profile = trajectoryProfile(trajectory)

    // Ensure beds remain unique even if upstream data repeats.
    let bed = patient.bed || `ICU-${index + 1}`
    if (takenBeds.has(bed)) {
      bed = randomPick(beds.filter((b) => b && !takenBeds.has(b))) || `ICU-${String(index + 1).padStart(2, '0')}`
    }
    takenBeds.add(bed)

    const baselineRisk = clamp(
      trajectory === 'severe' ? patient.risk + 18 : trajectory === 'recovery' ? patient.risk - 10 : patient.risk,
      8,
      95
    )

    return {
      ...patient,
      bed,
      risk: Math.round(baselineRisk),
      trend: '+0',
      lead: profile.lead,
      trajectory,
      waveform: [...patient.waveform.slice(0, -1), Math.round(baselineRisk)],
      history: [{ t: Date.now(), ...patient.vitals, risk: Math.round(baselineRisk) }],
    }
  })
}

function scoreContributions(vitals) {
  return (
    (vitals.HR - 85) * 0.24 +
    (92 - vitals.SpO2) * 1.7 +
    (vitals.Resp - 18) * 0.6 +
    (vitals.Temp - 37) * 4.5
  )
}

function statusForRisk(risk) {
  if (risk >= 85) return 'Critical'
  if (risk >= 70) return 'High'
  if (risk >= 45) return 'Watch'
  return 'Stable'
}

function updatePatient(patient, scenarioProfile, scenarioLead) {
  const trajectory = patient.trajectory || 'stable'
  const traj = trajectoryProfile(trajectory)
  const randomizer = Math.max(0.8, (scenarioProfile.volatility + traj.volatility) / 2)

  const combined = {
    hr: scenarioProfile.hr + traj.hr,
    spo2: scenarioProfile.spo2 + traj.spo2,
    resp: scenarioProfile.resp + traj.resp,
    temp: scenarioProfile.temp + traj.temp,
    riskDrift: scenarioProfile.riskDrift + traj.riskDrift,
  }

  const nextVitals = {
    HR: Math.round(clamp(patient.vitals.HR + combined.hr * 0.35 + randomCentered(randomizer), 45, 170)),
    SpO2: Math.round(clamp(patient.vitals.SpO2 + combined.spo2 * 0.25 + randomCentered(randomizer * 0.35), 75, 100)),
    Resp: Math.round(clamp(patient.vitals.Resp + combined.resp * 0.25 + randomCentered(randomizer * 0.4), 10, 42)),
    Temp: Number(clamp(patient.vitals.Temp + combined.temp * 0.12 + randomCentered(randomizer * 0.03), 34.5, 41).toFixed(1)),
  }

  const nextRisk = Math.round(
    clamp(
      patient.risk + combined.riskDrift * 0.35 + scoreContributions(nextVitals) * 0.05 + randomCentered(randomizer),
      8,
      99
    )
  )
  const riskChange = nextRisk - patient.risk
  // Compute lead: prefer an explicit deterioration label when risk is critical,
  // otherwise use trajectory-specific lead for recovering trajectories,
  // and fall back to the current scenario lead.
  let leadLabel = scenarioLead
  if (nextRisk >= 85) {
    leadLabel = 'Multi-organ deterioration'
  } else if (nextRisk < 45) {
    leadLabel = 'Baseline recovery'
  } else if (patient.trajectory === 'recovery') {
    leadLabel = traj.lead
  }

  return {
    ...patient,
    risk: nextRisk,
    trend: `${riskChange >= 0 ? '+' : ''}${riskChange}`,
    status: statusForRisk(nextRisk),
    lead: leadLabel,
    vitals: nextVitals,
    waveform: [...patient.waveform.slice(1), nextRisk],
  }
}

export function SimulationProvider({ children }) {
  const [activeScenario, setActiveScenario] = useState('baseline')
  const [patientQueue, setPatientQueue] = useState(() => seedIcuEnvironment(DEFAULT_PATIENTS))
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [isPaused, setIsPaused] = useState(true)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendError, setBackendError] = useState(null)
  const [backendVersion, setBackendVersion] = useState(null)
  // Data source: 'simulated' runs the local engine, 'live' mirrors the shared
  // backend scenario engine so every browser sees identical beds.
  const [source, setSource] = useState('simulated')
  const [liveMeta, setLiveMeta] = useState({ scenario: 'baseline', scenarioLabel: 'Baseline Mix', tick: 0, paused: true })
  const [liveError, setLiveError] = useState(null)
  const lastIngestTime = React.useRef({}) // Track last ingest time per patient
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

  // Synthetic display is only meaningful with a live backend. While the
  // backend is unreachable, stop advancing the beds as well as ingesting them.
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

  // Send vitals to backend for alert processing (with cooldown)
  useEffect(() => {
    if (isPaused || backendStatus !== 'online' || source !== 'simulated') return undefined

    const intervalId = window.setInterval(() => {
      if (document.visibilityState === 'hidden') return
      const { profile, lead } = SCENARIOS[activeScenario]
      setPatientQueue((current) => {
        const updated = current.map((patient) => {
          const next = updatePatient(patient, profile, lead)
          return {
            ...next,
            history: [...(patient.history || []), { t: Date.now(), ...next.vitals, risk: next.risk }].slice(-60),
          }
        })
        const now = Date.now()
        const INGEST_COOLDOWN_MS = 10000 // Send vitals max once every 10 seconds per patient
        
        // Send each patient's vitals to backend for risk scoring and alerts
        updated.forEach((patient) => {
          const lastTime = lastIngestTime.current[patient.patient_id] || 0
          
          // Only send if cooldown has passed
          if (now - lastTime >= INGEST_COOLDOWN_MS) {
            const vital = {
              patient_id: patient.patient_id,
              timestamp: Date.now() / 1000,
              HR: patient.vitals.HR,
              SpO2: patient.vitals.SpO2,
              RespRate: patient.vitals.Resp,
              Temp: patient.vitals.Temp,
            }
            
            fetch(`${API_URL}/ingest`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(vital),
            }).catch((err) => console.warn('Failed to ingest vital:', err))
            
            lastIngestTime.current[patient.patient_id] = now
          }
        })
        
        return updated
      })
      setLastUpdated(new Date())
    }, 1000)
    return () => window.clearInterval(intervalId)
  }, [activeScenario, backendStatus, isPaused, source])

  // Backend live mode: mirror the shared scenario engine. Local simulation
  // stays off while live so the two never fight over patientQueue.
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
      })
      setLiveError(null)
      setLastUpdated(new Date())
    } catch (err) {
      setLiveError(err?.message || 'Live control unreachable.')
    }
  }, [])

  const setDataSource = useCallback((next) => {
    setSource(next)
    setLiveError(null)
    if (next === 'simulated') {
      setPatientQueue(seedIcuEnvironment(DEFAULT_PATIENTS))
      setActiveScenario('baseline')
      setIsPaused(true)
      setLastUpdated(new Date())
    }
  }, [])

  const value = useMemo(
    () => {
      const live = source === 'live'
      return {
        activeScenario: live ? liveMeta.scenario : activeScenario,
        activeScenarioLabel: live ? liveMeta.scenarioLabel : SCENARIOS[activeScenario].label,
        scenarioEntries: Object.entries(SCENARIOS),
        patientQueue,
        lastUpdated,
        isPaused: live ? liveMeta.paused : isPaused,
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
        setActiveScenario: (key) => {
          if (live) controlLive('set_scenario', key)
          else setActiveScenario(key)
        },
        toggleSimulation: () => {
          if (live) controlLive(liveMeta.paused ? 'start' : 'pause')
          else setIsPaused((current) => !current)
        },
        resetSimulation: () => {
          if (live) {
            controlLive('reset')
            return
          }
          setPatientQueue(seedIcuEnvironment(DEFAULT_PATIENTS))
          setActiveScenario('baseline')
          setIsPaused(true)
          setLastUpdated(new Date())
        },
      }
    },
    [activeScenario, backendError, backendStatus, backendVersion, controlLive, liveError, liveMeta, patientQueue, lastUpdated, isPaused, reloadLive, retryBackend, source]
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
          <p className="muted">The synthetic stream is turned off while the backend is unreachable.</p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-primary btn-sm" onClick={retryBackend}>
            Retry connection
          </button>
        </div>
      </header>
      <p className="notice notice-error" role="alert">
        Backend unavailable. Live risk scores, model metrics, training jobs, and alert delivery need the
        backend, so this view is stopped instead of showing simulated patients.
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
