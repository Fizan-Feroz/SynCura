import React, { useRef } from 'react'
import { Link } from 'react-router-dom'
import { gsap, useGSAP, MOTION_OK } from '../motion/gsap'

const STAGES = [
  { name: 'Sources', detail: 'ESP32 + MAX30105 sensor, bedside monitors, or a replayed PhysioNet stay', tech: 'MQTT, HTTP' },
  { name: 'Ingest', detail: 'Validates each reading and hands it to the inference engine', tech: 'FastAPI /ingest, MQTT subscriber' },
  { name: 'Window', detail: 'Keeps the last 90 minutes of 12 features per patient and scales them with training statistics', tech: 'NumPy' },
  { name: 'Ensemble', detail: 'Three attention LSTMs score the window; their logits are averaged', tech: 'PyTorch' },
  { name: 'Outputs', detail: 'Risk 0 to 100 on the dashboard, stored readings, and alerts past a cooldown', tech: 'SQLite, Discord webhook' },
]

const STACK = [
  { group: 'Backend', items: [['FastAPI', 'Ingest, scores, metrics and training endpoints'], ['PyTorch', 'Loads the ensemble and runs inference on CPU'], ['SQLite', 'Stores every reading with its score'], ['paho-mqtt', 'Subscribes to device topics']] },
  { group: 'Model', items: [['Attention LSTM', '2 layers, hidden size 96, dropout 0.3, additive attention over time'], ['Ensemble', 'Three members; two trained on set A, one on set A plus 80% of set B'], ['SHAP', 'Per-feature explanations on request'], ['NEWS2', 'Rule-based baseline for comparison']] },
  { group: 'Frontend', items: [['React 18 + Vite', 'Single-page app, routes split into lazy chunks'], ['GSAP', 'Hero trace, bed re-ranking (Flip) and scroll progress'], ['Archivo', 'One variable typeface: wide headlines, narrow readouts']] },
  { group: 'Hardware', items: [['ESP32', 'Wi-Fi microcontroller publishing readings over MQTT'], ['MAX30105', 'Optical sensor for pulse and SpO2']] },
]

const LIMITS = [
  'No authentication, encryption or audit log. Run it on a trusted local network only.',
  'The label is in-hospital death from PhysioNet 2012, used as a stand-in for deterioration.',
  'The holdout set also guided which models joined the ensemble, so its AUC is slightly optimistic.',
  'Dashboard patients are simulated; their scores are not the model’s output.',
]

export default function ArchitecturePage() {
  const root = useRef(null)

  // One looping pulse travels the pipeline to show the direction of data.
  useGSAP(
    () => {
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        gsap.fromTo('.flow-pulse', { '--x': 0 }, { '--x': 1, duration: 3.2, ease: 'power1.inOut', repeat: -1, repeatDelay: 0.6 })
      })
    },
    { scope: root }
  )

  return (
    <div className="page arch" ref={root}>
      <header className="page-head">
        <div>
          <h1>Architecture</h1>
          <p className="lede">How a reading travels from a sensor to a risk score, what it runs on, and what this prototype does not do.</p>
        </div>
      </header>

      <section className="arch-section" aria-labelledby="flow-title">
        <h2 id="flow-title">From sensor to score</h2>
        <div className="flow">
          <div className="flow-rail" aria-hidden="true">
            <span className="flow-pulse" />
          </div>
          <ol className="flow-stages">
            {STAGES.map((s) => (
              <li key={s.name} className="flow-stage">
                <span className="flow-node" aria-hidden="true" />
                <h3>{s.name}</h3>
                <p>{s.detail}</p>
                <span className="flow-tech small muted">{s.tech}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="arch-section" aria-labelledby="stack-title">
        <h2 id="stack-title">What it runs on</h2>
        <div className="stack">
          {STACK.map((g) => (
            <div key={g.group} className="stack-group">
              <h3>{g.group}</h3>
              <dl>
                {g.items.map(([name, role]) => (
                  <div key={name}>
                    <dt>{name}</dt>
                    <dd>{role}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </section>

      <section className="arch-section arch-split" aria-labelledby="limits-title">
        <div>
          <h2 id="limits-title">Limits</h2>
          <ul className="caveats">
            {LIMITS.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </div>
        <div>
          <h2>Run it locally</h2>
          <p className="muted small">From the repository root, in two terminals:</p>
          <pre className="code"><code>{`python -m uvicorn backend.app:app --port 8000
cd frontend && npm run dev`}</code></pre>
          <p className="small"><Link to="/dashboard">Open the central station</Link> and start the stream.</p>
        </div>
      </section>
    </div>
  )
}
