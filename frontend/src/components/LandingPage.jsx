import React, { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { gsap, useGSAP, ScrollTrigger, MOTION_OK } from '../motion/gsap'
import { TopBar } from './AppShell'
import { BrandMark, Wordmark } from './Brand'
import HeroFilm from './film/HeroFilm'

const STEPS = [
  {
    title: 'Vitals arrive',
    body: 'Bedside monitors, the ESP32 pulse-oximeter or a replayed PhysioNet stay send readings over MQTT or HTTP. Each one joins that patient’s recent history.',
  },
  {
    title: 'A 90-minute window',
    body: 'The model looks at the last 90 minutes of twelve features: vital signs plus labs such as BUN, creatinine and platelets.',
  },
  {
    title: 'Scaled like training',
    body: 'Every value is normalised with the training-set statistics, so a heart rate of 120 means the same thing live as it did during training.',
  },
  {
    title: 'Three models agree',
    body: 'Three attention LSTMs score the window. Their outputs are averaged into one probability and shown as a risk from 0 to 100.',
  },
]

const METRICS = [
  { name: 'Holdout AUC', value: '0.844', fill: 0.844, note: '20% of PhysioNet set B' },
  { name: 'Validation AUC', value: '0.840', fill: 0.84, note: 'Patient-level split of set A' },
  { name: 'Recall', value: '80.7%', fill: 0.807, note: 'Deaths the model flagged' },
  { name: 'Accuracy', value: '74.7%', fill: 0.747, note: 'At a 0.5 threshold' },
  { name: 'Precision', value: '34.5%', fill: 0.345, note: 'Flags that were correct' },
]

const DESTINATIONS = [
  { to: '/dashboard', name: 'Central station', body: 'Twelve synthetic beds ranked by risk, with scenarios, alert threshold tuning and a NEWS2 comparison.' },
  { to: '/waveforms', name: 'Waveforms', body: 'Per-bed trends for heart rate, SpO2, respiration and temperature, drawn like monitor channels.' },
  { to: '/simulated-data', name: 'Data feed', body: 'The raw stream the dashboard sends to the backend, one row per bed.' },
  { to: '/training', name: 'Training', body: 'Start a training job against your PhysioNet files and watch loss and AUC as it runs.' },
  { to: '/architecture', name: 'Architecture', body: 'The pipeline from sensor to score, the stack, and the limits of this prototype.' },
]

export default function LandingPage({ theme, onToggleTheme }) {
  const root = useRef(null)

  // Deep links (e.g. /#how): the browser jumps before the pinned hero adds its
  // scroll length, landing mid-morph. Re-jump once layout is final.
  useEffect(() => {
    const id = window.location.hash.slice(1)
    if (!id) return undefined
    const t = window.setTimeout(() => {
      ScrollTrigger.refresh()
      document.getElementById(id)?.scrollIntoView()
    }, 120)
    return () => window.clearTimeout(t)
  }, [])

  // Scroll: the rail between the four steps fills as you read.
  useGSAP(
    () => {
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        gsap.fromTo(
          '.steps',
          { '--fill': 0 },
          {
            '--fill': 1,
            ease: 'none',
            scrollTrigger: { trigger: '.steps', start: 'top 80%', end: 'bottom 55%', scrub: 0.6 },
          }
        )
      })
    },
    { scope: root }
  )

  return (
    <div className="landing" ref={root}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <TopBar theme={theme} onToggleTheme={onToggleTheme} landing />

      <main id="main-content" tabIndex="-1">
        <HeroFilm />

        <section className="section how" id="how" aria-labelledby="how-title">
          <div className="section-head">
            <h2 id="how-title">How a score is made</h2>
            <p className="muted">Four steps run for every reading, from the sensor to the number on the screen.</p>
          </div>
          <ol className="steps">
            {STEPS.map((step, i) => (
              <li key={step.title} className="step">
                <span className="step-index num" aria-hidden="true">{i + 1}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
        </section>

        <section className="section evidence" id="evidence" aria-labelledby="evidence-title">
          <div className="evidence-copy">
            <h2 id="evidence-title">How well it works, and where it doesn’t</h2>
            <p className="muted">
              Trained and evaluated on the PhysioNet 2012 challenge: 8,000 ICU stays across sets A and B.
              The label is in-hospital death, used here as a stand-in for deterioration.
            </p>
            <ul className="caveats">
              <li>The holdout was not used to train weights, but it was used to choose the three ensemble members, so it is not a locked test set.</li>
              <li>Precision is low. At the default threshold, about two in three alerts would be false alarms.</li>
              <li>This is not a validated clinical tool and must not guide patient care.</li>
            </ul>
          </div>
          <table className="metrics">
            <caption className="sr-only">Deployed ensemble metrics</caption>
            <thead>
              <tr>
                <th scope="col">Measure</th>
                <th scope="col">Value</th>
                <th scope="col"><span className="sr-only">Scale</span></th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => (
                <tr key={m.name}>
                  <th scope="row">
                    {m.name}
                    <span className="metric-note">{m.note}</span>
                  </th>
                  <td className="num metric-value">{m.value}</td>
                  <td className="metric-bar" aria-hidden="true">
                    <span style={{ '--v': m.fill }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="section inside" aria-labelledby="inside-title">
          <div className="section-head">
            <h2 id="inside-title">Inside the prototype</h2>
            <p className="muted">Every screen runs on synthetic patients, so you can try it without clinical data.</p>
          </div>
          <ul className="destinations">
            {DESTINATIONS.map((d) => (
              <li key={d.to}>
                <Link to={d.to} className="destination">
                  <span className="destination-name">{d.name}</span>
                  <span className="destination-body">{d.body}</span>
                  <svg className="destination-glyph" viewBox="0 0 60 20" aria-hidden="true">
                    <path d="M0 10h16l3-6 4 12 4-14 4 8h29" />
                  </svg>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      </main>

      <footer className="site-footer">
        <div className="site-footer-brand">
          <BrandMark />
          <Wordmark />
        </div>
        <p className="small muted">
          Research prototype built on the PhysioNet 2012 challenge dataset. Not for clinical use.
        </p>
      </footer>
    </div>
  )
}
