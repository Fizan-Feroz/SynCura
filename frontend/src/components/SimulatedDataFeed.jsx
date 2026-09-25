import React from 'react'
import { useSimulation } from '../simulationContext'
import { riskLabel, riskTone } from './trace'

export default function SimulatedDataFeed() {
  const { activeScenarioLabel, patientQueue, lastUpdated, isPaused, toggleSimulation, resetSimulation } = useSimulation()

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Data feed</h1>
          <p className="muted">
            The readings the dashboard generates and sends to <code>/ingest</code>, one row per bed.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-primary btn-sm" onClick={toggleSimulation}>
            {isPaused ? 'Start stream' : 'Pause stream'}
          </button>
          <button type="button" className="btn btn-quiet btn-sm" onClick={resetSimulation}>
            Reset beds
          </button>
        </div>
      </header>

      <dl className="feed-meta">
        <div><dt>Scenario</dt><dd>{activeScenarioLabel}</dd></div>
        <div><dt>Stream</dt><dd>{isPaused ? 'Paused' : 'Running'}</dd></div>
        <div><dt>Last update</dt><dd className="num">{lastUpdated.toLocaleTimeString()}</dd></div>
      </dl>

      <div className="table-wrap" tabIndex="0" aria-label="Simulated patient stream, scrollable">
        <table className="feed-table">
          <thead>
            <tr>
              <th scope="col">Bed</th>
              <th scope="col">Patient</th>
              <th scope="col">Status</th>
              <th scope="col" className="r">Risk</th>
              <th scope="col" className="r">Change</th>
              <th scope="col" className="r" style={{ '--c': 'var(--hr)' }}>HR</th>
              <th scope="col" className="r" style={{ '--c': 'var(--spo2)' }}>SpO2</th>
              <th scope="col" className="r" style={{ '--c': 'var(--rr)' }}>RR</th>
              <th scope="col" className="r" style={{ '--c': 'var(--temp)' }}>Temp</th>
              <th scope="col">Signal</th>
              <th scope="col">Last six risks</th>
            </tr>
          </thead>
          <tbody>
            {patientQueue.map((p) => {
              const tone = riskTone(p.risk)
              return (
                <tr key={p.patient_id}>
                  <th scope="row">{p.bed}</th>
                  <td className="num">{p.patient_id}</td>
                  <td><span className={`status status-${tone}`}>{riskLabel(p.risk)}</span></td>
                  <td className="num r strong">{p.risk}%</td>
                  <td className="num r">{p.trend}</td>
                  <td className="num r">{p.vitals.HR}</td>
                  <td className="num r">{p.vitals.SpO2}</td>
                  <td className="num r">{p.vitals.Resp}</td>
                  <td className="num r">{p.vitals.Temp.toFixed(1)}</td>
                  <td>{p.lead}</td>
                  <td className="num muted">{p.waveform.slice(-6).join('  ')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
