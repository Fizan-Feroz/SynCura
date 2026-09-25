import React, { useMemo, useState } from 'react'
import { useSimulation } from '../simulationContext'
import { riskLabel, riskTone, seriesPath } from './trace'

const CHANNELS = {
  HR: { label: 'Heart rate', short: 'HR', unit: 'bpm', color: 'var(--hr)' },
  SpO2: { label: 'SpO2', short: 'SpO2', unit: '%', color: 'var(--spo2)' },
  Resp: { label: 'Respiration', short: 'RR', unit: '/min', color: 'var(--rr)' },
  Temp: { label: 'Temperature', short: 'Temp', unit: '°C', color: 'var(--temp)' },
}
const ORDER = ['HR', 'SpO2', 'Resp', 'Temp']

export default function SensorWaveform() {
  const { patientQueue } = useSimulation()
  const [shown, setShown] = useState({ HR: true, SpO2: true, Resp: true, Temp: true })

  const rows = useMemo(
    () =>
      patientQueue.map((patient) => ({
        patient,
        series: {
          HR: patient.waveform.map((p, i) => p + Math.sin(i / 2) * 5),
          SpO2: patient.waveform.map((p, i) => 88 + (p - 50) * 0.18 - Math.cos(i / 3) * 1.1),
          Resp: patient.waveform.map((p, i) => 14 + (p - 50) * 0.1 + Math.sin(i / 4) * 0.8),
          Temp: patient.waveform.map((p, i) => 36.2 + (p - 50) * 0.03 + Math.cos(i / 5) * 0.03),
        },
      })),
    [patientQueue]
  )
  const channels = ORDER.filter((k) => shown[k])

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Waveforms</h1>
          <p className="muted">Recent trends for every bed, drawn like monitor channels. Synthetic data.</p>
        </div>
        <div className="channel-toggles" role="group" aria-label="Channels">
          {ORDER.map((k) => (
            <button
              key={k}
              type="button"
              className="channel-toggle"
              style={{ '--c': CHANNELS[k].color }}
              aria-pressed={shown[k]}
              onClick={() => setShown((s) => ({ ...s, [k]: !s[k] }))}
            >
              {CHANNELS[k].label}
            </button>
          ))}
        </div>
      </header>

      <div className="monitors">
        {rows.map(({ patient, series }) => {
          const tone = riskTone(patient.risk)
          return (
            <article key={patient.patient_id} className={`monitor tone-${tone}`} aria-label={`${patient.bed} monitor`}>
              <header className="monitor-head">
                <strong>{patient.bed}</strong>
                <span className="muted small">Patient {patient.patient_id}</span>
                <span className={`status status-${tone}`}>{riskLabel(patient.risk)}</span>
                <span className="monitor-risk num">Risk {patient.risk}</span>
              </header>
              {channels.length === 0 ? (
                <p className="muted small monitor-empty">Turn on a channel above to see its trace.</p>
              ) : (
                <div className="monitor-channels">
                  {channels.map((k) => (
                    <div key={k} className="channel" style={{ '--c': CHANNELS[k].color }}>
                      <svg viewBox="0 0 300 48" preserveAspectRatio="none" role="img" aria-label={`${CHANNELS[k].label} trend for ${patient.bed}`}>
                        <path d={seriesPath(series[k], 300, 48, 4)} vectorEffect="non-scaling-stroke" />
                      </svg>
                      <span className="channel-read">
                        <span className="channel-label">{CHANNELS[k].short}</span>
                        <span className="num channel-value">
                          {k === 'Temp' ? patient.vitals[k].toFixed(1) : patient.vitals[k]}
                          <small>{CHANNELS[k].unit}</small>
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
