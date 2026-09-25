import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { API_URL } from '../api'

const SERVING_FEATURES = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2', 'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']
const FEATURE_GROUPS = [
  { name: 'Vital signs', items: ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2', 'EtCO2'] },
  { name: 'Neuro and labs', items: ['GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose'] },
]

export default function TrainingConfig() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [form, setForm] = useState({
    physionet_path: '',
    outcomes_path: '',
    epochs: 5,
    batch_size: 32,
    learning_rate: 0.001,
    max_patients: 100,
    vital_features: SERVING_FEATURES,
  })

  const onChange = (e) => {
    const { name, value, type } = e.target
    setForm((f) => ({ ...f, [name]: type === 'number' ? parseFloat(value) : value }))
  }
  const toggleFeature = (feature) =>
    setForm((f) => ({
      ...f,
      vital_features: f.vital_features.includes(feature)
        ? f.vital_features.filter((x) => x !== feature)
        : [...f.vital_features, feature],
    }))

  const servable =
    form.vital_features.length === SERVING_FEATURES.length && SERVING_FEATURES.every((x) => form.vital_features.includes(x))

  const onSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await axios.post(`${API_URL}/training/start`, form)
      navigate(`/training/${res.data.job_id}`)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Training could not start.')
      setLoading(false)
    }
  }

  return (
    <div className="page page-narrow">
      <header className="page-head">
        <div>
          <Link to="/training" className="back-link">Training jobs</Link>
          <h1>New training job</h1>
          <p className="muted">Train an attention LSTM on your PhysioNet 2012 files. The result is saved as its own artifact and never replaces the deployed model.</p>
        </div>
      </header>

      {error && <p className="notice notice-error" role="alert">{error}</p>}

      <form className="form" onSubmit={onSubmit}>
        <fieldset className="form-section">
          <legend>Data</legend>
          <div className="field">
            <label htmlFor="tc-physionet">PhysioNet folder</label>
            <input id="tc-physionet" type="text" name="physionet_path" value={form.physionet_path} onChange={onChange} placeholder="C:\data\set-a" required />
            <span className="hint">The set-a (or set-b) folder with one .txt file per patient.</span>
          </div>
          <div className="field">
            <label htmlFor="tc-outcomes">Outcomes file</label>
            <input id="tc-outcomes" type="text" name="outcomes_path" value={form.outcomes_path} onChange={onChange} placeholder="C:\data\Outcomes-a.txt" required />
            <span className="hint">Provides the in-hospital death label for each patient.</span>
          </div>
        </fieldset>

        <fieldset className="form-section">
          <legend>Training</legend>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="tc-epochs">Epochs</label>
              <input id="tc-epochs" type="number" name="epochs" value={form.epochs} onChange={onChange} min="1" max="100" />
            </div>
            <div className="field">
              <label htmlFor="tc-batch">Batch size</label>
              <input id="tc-batch" type="number" name="batch_size" value={form.batch_size} onChange={onChange} min="1" max="256" />
            </div>
            <div className="field">
              <label htmlFor="tc-lr">Learning rate</label>
              <input id="tc-lr" type="number" name="learning_rate" value={form.learning_rate} onChange={onChange} min="0.00001" max="0.1" step="0.0001" />
            </div>
            <div className="field">
              <label htmlFor="tc-max">Patients to load</label>
              <input id="tc-max" type="number" name="max_patients" value={form.max_patients} onChange={onChange} min="10" max="10000" />
            </div>
          </div>
        </fieldset>

        <fieldset className="form-section">
          <legend>Features</legend>
          {FEATURE_GROUPS.map((g) => (
            <div key={g.name} className="feature-group">
              <span className="field-label">{g.name}</span>
              <div className="feature-chips">
                {g.items.map((f) => (
                  <label key={f} className="chip">
                    <input type="checkbox" checked={form.vital_features.includes(f)} onChange={() => toggleFeature(f)} />
                    <span>{f}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
          <p className={`small ${servable ? 'muted' : 'feature-warning'}`}>
            {servable
              ? 'These twelve features match the deployed model, so this run could later be promoted to serving.'
              : 'This feature set differs from the deployed model. The run will train, but the backend cannot serve it.'}
          </p>
        </fieldset>

        <div className="form-actions">
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Starting training…' : 'Start training'}
          </button>
          <Link to="/training" className="btn btn-ghost">Cancel</Link>
        </div>
      </form>
    </div>
  )
}
