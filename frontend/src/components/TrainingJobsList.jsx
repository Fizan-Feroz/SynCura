import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { API_URL } from '../api'

export function JobStatus({ status }) {
  const tone = { pending: 'watch', running: 'info', completed: 'stable', failed: 'critical' }[status] || 'watch'
  return <span className={`job-status job-${tone}`} role="status">{status}</span>
}

export default function TrainingJobsList() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      try {
        const res = await axios.get(`${API_URL}/training/jobs`)
        setJobs([...res.data.jobs].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at))))
        setError(null)
      } catch (err) {
        setError(err.response?.data?.detail || err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Training jobs</h1>
          <p className="muted">Each job trains in the background and saves its own model file. Refreshes every three seconds.</p>
        </div>
        <div className="page-actions">
          <Link to="/training/new" className="btn btn-primary btn-sm">New training job</Link>
        </div>
      </header>

      {error && (
        <p className="notice notice-error" role="alert">
          Could not reach the backend ({error}). Start it with <code>uvicorn backend.app:app</code> and this list will load.
        </p>
      )}

      {loading ? (
        <div className="skeleton" role="status"><span className="sr-only">Loading jobs</span></div>
      ) : jobs.length === 0 && !error ? (
        <div className="empty">
          <h2>No training jobs yet</h2>
          <p className="muted">Point a job at your PhysioNet files to train a model and watch its metrics arrive.</p>
          <Link to="/training/new" className="btn btn-primary">Start the first job</Link>
        </div>
      ) : (
        <ul className="jobs">
          {jobs.map((job) => {
            const progress = job.total_epochs > 0 ? Math.round((job.current_epoch / job.total_epochs) * 100) : 0
            const m = job.metrics || {}
            return (
              <li key={job.job_id}>
                <Link to={`/training/${job.job_id}`} className="job">
                  <span className="job-main">
                    <span className="job-name">{job.job_id.replace(/^job_/, '')}</span>
                    <span className="muted small">
                      {job.created_at ? new Date(job.created_at).toLocaleString() : 'Unknown start'}
                    </span>
                  </span>
                  <JobStatus status={job.status} />
                  <span className="job-progress" aria-label={`Epoch ${job.current_epoch} of ${job.total_epochs}`}>
                    <span className="num">{job.current_epoch}/{job.total_epochs}</span>
                    <span className="progress"><span style={{ '--p': progress / 100 }} /></span>
                  </span>
                  <span className="job-metrics num">
                    {m.auc != null && <span>AUC {m.auc.toFixed(3)}</span>}
                    {m.train_loss != null && <span>Loss {m.train_loss.toFixed(4)}</span>}
                  </span>
                  {job.error_message && <span className="job-error small">{job.error_message}</span>}
                </Link>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
