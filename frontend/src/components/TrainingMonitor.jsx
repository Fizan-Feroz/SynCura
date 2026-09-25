import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import axios from 'axios'
import { API_URL } from '../api'
import { JobStatus } from './TrainingJobsList'

const METRIC_LABELS = [
  ['train_loss', 'Training loss', (v) => v.toFixed(4)],
  ['auc', 'AUC', (v) => v.toFixed(4)],
  ['accuracy', 'Accuracy', (v) => `${(v * 100).toFixed(1)}%`],
  ['precision', 'Precision', (v) => `${(v * 100).toFixed(1)}%`],
  ['recall', 'Recall', (v) => `${(v * 100).toFixed(1)}%`],
  ['val_accuracy', 'Validation accuracy', (v) => `${(v * 100).toFixed(1)}%`],
]

export default function TrainingMonitor() {
  const { jobId } = useParams()
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    let timer
    const load = async () => {
      try {
        const res = await axios.get(`${API_URL}/training/${jobId}`)
        if (!alive) return
        setJob(res.data)
        setError(null)
        if (res.data.status !== 'completed' && res.data.status !== 'failed') timer = setTimeout(load, 2000)
      } catch (err) {
        if (alive) setError(err.response?.data?.detail || err.message)
      }
    }
    load()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [jobId])

  if (error) {
    return (
      <div className="page page-narrow">
        <Link to="/training" className="back-link">Training jobs</Link>
        <h1>Job not available</h1>
        <p className="notice notice-error" role="alert">{error}</p>
      </div>
    )
  }

  if (!job) {
    return <div className="page page-narrow"><div className="skeleton" role="status"><span className="sr-only">Loading job</span></div></div>
  }

  const progress = job.total_epochs > 0 ? Math.round((job.current_epoch / job.total_epochs) * 100) : 0
  const metrics = METRIC_LABELS.filter(([k]) => job.metrics?.[k] != null)

  return (
    <div className="page page-narrow">
      <header className="page-head">
        <div>
          <Link to="/training" className="back-link">Training jobs</Link>
          <h1 className="job-title">{jobId.replace(/^job_/, '')}</h1>
        </div>
        <JobStatus status={job.status} />
      </header>

      <section className="run-progress" aria-label="Progress">
        <div className="run-progress-head">
          <span>Epoch <strong className="num">{job.current_epoch}</strong> of <span className="num">{job.total_epochs}</span></span>
          <span className="num">{progress}%</span>
        </div>
        <div className="progress progress-lg" role="progressbar" aria-valuenow={progress} aria-valuemin="0" aria-valuemax="100" aria-label="Training progress">
          <span style={{ '--p': progress / 100 }} />
        </div>
      </section>

      {job.error_message && <p className="notice notice-error" role="alert">{job.error_message}</p>}

      <section className="run-section" aria-labelledby="metrics-title">
        <h2 id="metrics-title">Metrics</h2>
        {metrics.length === 0 ? (
          <p className="muted">Metrics appear after the first epoch finishes.</p>
        ) : (
          <dl className="run-metrics">
            {metrics.map(([k, label, f]) => (
              <div key={k}>
                <dt>{label}</dt>
                <dd className="num">{f(job.metrics[k])}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="run-section" aria-labelledby="config-title">
        <h2 id="config-title">Configuration</h2>
        <dl className="config">
          {job.config &&
            Object.entries(job.config).map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{Array.isArray(v) ? v.join(', ') : typeof v === 'object' ? JSON.stringify(v) : String(v)}</dd>
              </div>
            ))}
        </dl>
      </section>
    </div>
  )
}
