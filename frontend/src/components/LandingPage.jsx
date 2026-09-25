import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { gsap, useGSAP, MOTION_OK } from '../motion/gsap'
import { TopBar } from './AppShell'
import { BrandMark, Wordmark } from './Brand'
import { ecgPath } from './trace'

// Hero geometry (SVG user units; the SVG stretches to full width).
const W = 1440
const H = 260
const BASE = 150
const TRACE_END = 1000
const FORECAST = `M${TRACE_END} ${BASE} C1090 ${BASE} 1130 ${BASE - 6} 1190 ${BASE - 40} S1300 70 1360 52`
const RISK = 64

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

function useDriftingVitals() {
  const [v, setV] = useState({ HR: 88, SpO2: 95, RR: 21, Temp: 37.6 })
  useEffect(() => {
    const id = window.setInterval(() => {
      if (document.visibilityState === 'hidden') return
      setV((c) => ({
        HR: Math.max(80, Math.min(104, c.HR + Math.round(Math.random() * 4 - 1.6))),
        SpO2: Math.max(91, Math.min(97, c.SpO2 + (Math.random() > 0.7 ? -1 : Math.random() > 0.85 ? 1 : 0))),
        RR: Math.max(18, Math.min(26, c.RR + (Math.random() > 0.75 ? 1 : Math.random() > 0.85 ? -1 : 0))),
        Temp: Number(Math.max(37.2, Math.min(38.4, c.Temp + (Math.random() - 0.45) * 0.1)).toFixed(1)),
      }))
    }, 2200)
    return () => window.clearInterval(id)
  }, [])
  return v
}

export default function LandingPage({ theme, onToggleTheme }) {
  const root = useRef(null)
  const riskRef = useRef(null)
  const vitals = useDriftingVitals()
  const trace = useMemo(() => ecgPath({ x0: 0, x1: TRACE_END, base: BASE, amp: 92, beats: 7 }), [])

  useGSAP(
    () => {
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        const path = root.current.querySelector('.hero-trace-line')
        const pen = root.current.querySelector('.hero-pen')
        const len = path.getTotalLength()

        // x -> y lookup so the pen can ride the sweep head.
        const samples = Array.from({ length: 600 }, (_, i) => path.getPointAtLength((i / 599) * len))
        const yAt = (x) => {
          let lo = 0
          let hi = samples.length - 1
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1
            if (samples[mid].x < x) lo = mid
            else hi = mid
          }
          return samples[hi].y
        }
        const placePen = (x, y) => gsap.set(pen, { left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%` })

        gsap.set(path, { strokeDasharray: len, strokeDashoffset: len })
        gsap.set('.forecast-reveal', { attr: { width: 0 } })
        gsap.set(pen, { autoAlpha: 1 })

        const counter = { v: 0 }
        const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })
        tl.from('.hero-title .line > span', { yPercent: 115, duration: 1.1, stagger: 0.09 })
          .to(
            path,
            {
              strokeDashoffset: 0,
              duration: 2.6,
              ease: 'none',
              onUpdate() {
                const p = path.getPointAtLength(len * this.progress())
                placePen(p.x, p.y)
              },
            },
            0.25
          )
          .to('.forecast-reveal', { attr: { width: W - TRACE_END }, duration: 1.3, ease: 'power2.inOut' })
          .from('.risk-readout', { autoAlpha: 0, y: 10, duration: 0.7 }, '<0.55')
          .to(
            counter,
            {
              v: RISK,
              duration: 1.4,
              ease: 'power2.out',
              onUpdate: () => {
                if (riskRef.current) riskRef.current.textContent = Math.round(counter.v)
              },
            },
            '<'
          )
          .from('.hero-foot > *', { autoAlpha: 0, y: 18, duration: 0.8, stagger: 0.1 }, 0.9)

        // After the first draw: monitor-style refresh. A gap erases just
        // ahead of the write head, which rides the trace.
        tl.add(() => {
          const gap = root.current.querySelector('.sweep-gap')
          gsap.set(gap, { autoAlpha: 1 })
          const head = { x: 0 }
          gsap.to(head, {
            x: TRACE_END,
            duration: 5.2,
            ease: 'none',
            repeat: -1,
            onUpdate: () => {
              gsap.set(gap, { attr: { x: head.x + 6 } })
              placePen(head.x, yAt(head.x))
            },
          })
        })
      })
    },
    { scope: root }
  )

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
        <header className="hero grid-paper">
          <div className="hero-copy">
            <h1 className="hero-title">
              <span className="line"><span>An early warning,</span></span>
              <span className="line"><span>read from the last</span></span>
              <span className="line"><span>90 minutes of vitals.</span></span>
            </h1>
          </div>

          <div className="hero-trace">
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <clipPath id="forecast-clip">
                  <rect className="forecast-reveal" x={TRACE_END} y="0" width={W - TRACE_END} height={H} />
                </clipPath>
                <mask id="sweep-mask" maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
                  <rect x="0" y="0" width={W} height={H} fill="#fff" />
                  <rect className="sweep-gap" x="-60" y="0" width="34" height={H} fill="#000" opacity="0" />
                </mask>
                <linearGradient id="forecast-band" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="var(--watch)" stopOpacity="0.16" />
                  <stop offset="55%" stopColor="var(--watch)" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path className="hero-trace-line" d={trace} mask="url(#sweep-mask)" vectorEffect="non-scaling-stroke" />
              <g clipPath="url(#forecast-clip)">
                <path className="forecast-band" d={`${FORECAST} L1360 ${H} L${TRACE_END} ${H} Z`} fill="url(#forecast-band)" />
                <path className="forecast-line" d={FORECAST} vectorEffect="non-scaling-stroke" />
              </g>
              <line className="now-line" x1={TRACE_END} x2={TRACE_END} y1="18" y2={H - 12} vectorEffect="non-scaling-stroke" />
            </svg>
            <span className="hero-pen" aria-hidden="true" />

            <div className="now-label small">now</div>
            <div className="risk-readout" role="img" aria-label={`Example deterioration risk ${RISK} percent`}>
              <span className="risk-readout-label">Deterioration risk</span>
              <span className="risk-readout-value num">
                <span ref={riskRef}>{RISK}</span>
                <small>%</small>
              </span>
              <span className="risk-readout-note">Synthetic example</span>
            </div>
          </div>

          <div className="hero-foot">
            <p className="lede">
              SynCura watches ICU patients the way a nurse scans a monitor, then asks a trained model how
              likely each one is to deteriorate. It is a research prototype built on PhysioNet ICU data.
            </p>
            <div className="hero-actions">
              <Link to="/dashboard" className="btn btn-primary">Open the central station</Link>
              <a href="#how" className="btn btn-ghost">How a score is made</a>
            </div>
            <dl className="hero-vitals" aria-label="Example patient vitals (synthetic)">
              <div style={{ '--c': 'var(--hr)' }}><dt>HR</dt><dd className="num">{vitals.HR}</dd></div>
              <div style={{ '--c': 'var(--spo2)' }}><dt>SpO2</dt><dd className="num">{vitals.SpO2}<small>%</small></dd></div>
              <div style={{ '--c': 'var(--rr)' }}><dt>RR</dt><dd className="num">{vitals.RR}</dd></div>
              <div style={{ '--c': 'var(--temp)' }}><dt>Temp</dt><dd className="num">{vitals.Temp.toFixed(1)}<small>°C</small></dd></div>
            </dl>
          </div>
        </header>

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
