import React, { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { gsap, useGSAP, ScrollTrigger, MOTION_OK } from '../../motion/gsap'
import { SCENES, W, H, CX, CY, R, paintScene, rimEcg } from './scenes'

const SEEN_KEY = 'syncura-film-seen'
const RISK = 64
const FINAL_DROP = 110 // the final horizon sits a little lower than the film's
const FINAL_Y = CY + FINAL_DROP
const WORD_ARC = `M ${CX - (R + 16)} ${CY} A ${R + 16} ${R + 16} 0 0 1 ${CX + R + 16} ${CY}`
const toPath = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')

function readSeen() {
  try {
    return sessionStorage.getItem(SEEN_KEY) === '1'
  } catch {
    return false
  }
}

function markSeen() {
  try {
    sessionStorage.setItem(SEEN_KEY, '1')
  } catch {
    /* storage unavailable */
  }
}

export default function HeroFilm() {
  const root = useRef(null)
  const canvases = useRef([])
  const film = useRef(null)
  const ambient = useRef(null)
  const [state, setState] = useState('film') // film | done
  const [paused, setPaused] = useState(false)
  const rimPath = useMemo(() => toPath(rimEcg({ cx: 0, cy: 0, r: R - 4, amp: 58, beats: 12, span: 0.72, steps: 1400 })), [])

  // The stage fills the viewport below the sticky top bar (taller on phones).
  useLayoutEffect(() => {
    const bar = document.querySelector('.topbar')
    if (!bar) return undefined
    const sync = () => {
      root.current?.style.setProperty('--topbar-h', `${bar.offsetHeight}px`)
      ScrollTrigger.refresh()
    }
    sync()
    const ro = new ResizeObserver(sync)
    ro.observe(bar)
    return () => ro.disconnect()
  }, [])

  // Paint the first frame synchronously, the rest just behind it.
  useLayoutEffect(() => {
    paintScene(canvases.current[0], SCENES[0])
    const ids = SCENES.slice(1).map((scene, i) =>
      window.setTimeout(() => paintScene(canvases.current[i + 1], scene), 40 * (i + 1))
    )
    return () => ids.forEach(window.clearTimeout)
  }, [])

  useGSAP(
    (context, contextSafe) => {
      const q = gsap.utils.selector(root)
      const world = { x: CX, y: FINAL_Y, s: 1 }
      const worldEl = q('.final-world')[0]
      const applyWorld = () => worldEl.setAttribute('transform', `translate(${world.x} ${world.y}) scale(${world.s})`)
      const word = q('.film-word')[0]
      const wordPath = q('.film-word textPath')[0]
      const ecgPath = q('.final-ecg')[0]
      const pen = q('.final-pen')[0]
      const ecgLen = ecgPath.getTotalLength()

      const finish = contextSafe(() => {
        markSeen()
        setState('done')
      })

      // Ambient: a write head rides the rim trace, like a monitor sweep.
      const startAmbient = () => {
        ambient.current?.kill()
        const head = { p: 0 }
        ambient.current = gsap.to(head, {
          p: 1,
          duration: 7,
          ease: 'none',
          repeat: -1,
          onUpdate: () => {
            const pt = ecgPath.getPointAtLength(head.p * ecgLen)
            pen.setAttribute('cx', pt.x)
            pen.setAttribute('cy', pt.y)
          },
        })
      }

      const mm = gsap.matchMedia()

      mm.add(MOTION_OK, () => {
        // ---------------- the film ----------------
        const layers = q('.film-layer')
        gsap.set(layers, { autoAlpha: 0 })
        gsap.set(['.film-final', '.film-copy > *', word], { autoAlpha: 0 })
        gsap.set(ecgPath, { strokeDasharray: ecgLen, strokeDashoffset: ecgLen })
        gsap.set(pen, { autoAlpha: 0 })

        const tl = gsap.timeline({
          paused: true,
          onUpdate() {
            root.current?.style.setProperty('--film-progress', this.progress())
          },
          onComplete: () => {
            finish()
            startAmbient()
          },
        })

        let t = 0
        SCENES.forEach((scene, i) => {
          const layer = layers[i]
          tl.set(layers, { autoAlpha: 0 }, t)
            .set(layer, { autoAlpha: 1 }, t)
            .call(() => {
              wordPath.textContent = scene.word
              word.setAttribute('fill', scene.ink)
            }, null, t)
            .fromTo(layer, { scale: 1.08 }, { scale: 1, duration: scene.dur + 0.15, ease: 'none' }, t)
          if (i === 0) {
            tl.fromTo(layer, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.5, ease: 'power1.in' }, 0)
              .fromTo(word, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.4 }, 0.35)
          }
          t += scene.dur
        })

        // Cut to black, then the SynCura horizon rises.
        tl.set([layers, word], { autoAlpha: 0 }, t)
        t += 0.35
        tl.set('.film-final', { autoAlpha: 1 }, t)
          .fromTo(world, { y: FINAL_Y + 240 }, { y: FINAL_Y, duration: 1.6, ease: 'power3.out', onUpdate: applyWorld }, t)
          .fromTo('.final-halo', { opacity: 0 }, { opacity: 1, duration: 1.4, ease: 'power2.out' }, t)
          .to(ecgPath, { strokeDashoffset: 0, duration: 1.8, ease: 'power2.inOut' }, t + 0.4)
          .fromTo('.final-word', { autoAlpha: 0, y: 14 }, { autoAlpha: 1, y: 0, duration: 0.9, ease: 'power2.out' }, t + 0.7)
          .set(pen, { autoAlpha: 1 }, t + 2.2)
          .fromTo('.film-copy > *', { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: 0.8, stagger: 0.1, ease: 'power3.out' }, t + 1.5)

        film.current = tl
        if (readSeen()) tl.progress(1)
        else tl.play(0)

        // ---------------- scroll: horizon -> risk ring ----------------
        const ringTarget = () => {
          const stage = root.current.getBoundingClientRect()
          const slot = q('.ring-slot')[0].getBoundingClientRect()
          const k = Math.max(stage.width / W, stage.height / H) // slice scale
          const ox = (stage.width - W * k) / 2
          const oy = (stage.height - H * k) / 2
          const toX = (px) => (px - stage.left - ox) / k
          const toY = (py) => (py - stage.top - oy) / k
          const cx = toX(slot.left + slot.width / 2)
          const cy = toY(slot.top + slot.height / 2)
          const r = (slot.width / 2) / k - 10
          return { x: cx, y: cy, s: r / R, r }
        }
        const placeRing = () => {
          const { x, y, r } = ringTarget()
          q('.ring').forEach((g) => g.setAttribute('transform', `translate(${x} ${y}) rotate(-90)`))
          q('.ring circle').forEach((c) => c.setAttribute('r', r))
        }
        placeRing()

        const counter = { v: 0 }
        const st = gsap.timeline({
          // Start states come from CSS / the film; don't stamp them at creation.
          defaults: { immediateRender: false },
          scrollTrigger: {
            trigger: root.current,
            start: () => `top ${document.querySelector('.topbar')?.offsetHeight || 0}px`,
            end: '+=170%',
            pin: true,
            scrub: 0.8,
            invalidateOnRefresh: true,
            onRefresh: placeRing,
            onUpdate: (self) => {
              if (self.progress > 0.002 && tl.progress() < 1) tl.progress(1)
            },
          },
        })
        st.fromTo(['.film-copy', '.film-controls'], { autoAlpha: 1, y: 0 }, { autoAlpha: 0, y: -40, duration: 0.16 }, 0)
          .fromTo('.final-word', { autoAlpha: 1 }, { autoAlpha: 0, duration: 0.1 }, 0)
          .fromTo(world, { x: CX, y: FINAL_Y, s: 1 }, {
            x: () => ringTarget().x,
            y: () => ringTarget().y,
            s: () => ringTarget().s,
            duration: 0.42,
            ease: 'power2.inOut',
            onUpdate: applyWorld,
          }, 0.04)
          .fromTo(['.final-ecg', '.final-pen'], { opacity: 1 }, { opacity: 0, duration: 0.15 }, 0.28)
          .fromTo('.final-halo', { opacity: 1 }, { opacity: 0, duration: 0.25 }, 0.32)
          .fromTo('.film-paper', { opacity: 0 }, { opacity: 1, duration: 0.1 }, 0.46)
          .fromTo(['.final-disc', '.final-rim'], { opacity: 1 }, { opacity: 0, duration: 0.08 }, 0.47)
          .fromTo('.ring', { opacity: 0 }, { opacity: 1, duration: 0.08 }, 0.47)
          .fromTo('.ring-value', { strokeDashoffset: 100 }, { strokeDashoffset: 100 - RISK, duration: 0.28, ease: 'power2.out' }, 0.52)
          .to(counter, {
            v: RISK,
            duration: 0.3,
            ease: 'power2.out',
            onUpdate: () => {
              const el = q('.ring-num-value')[0]
              if (el) el.textContent = Math.round(counter.v)
            },
          }, 0.52)
          .fromTo('.ring-copy > *', { autoAlpha: 0, y: 26 }, { autoAlpha: 1, y: 0, duration: 0.2, stagger: 0.04 }, 0.58)
          .fromTo('.ring-num', { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.1 }, 0.52)
          .to({}, { duration: 0.2 }) // hold the finished ring before unpinning

        ScrollTrigger.refresh()
      })

      mm.add('(prefers-reduced-motion: reduce)', () => {
        // Final frame, no film, no scroll-jacking.
        gsap.set('.film-layer', { autoAlpha: 0 })
        gsap.set('.film-word', { autoAlpha: 0 })
        gsap.set(ecgPath, { strokeDashoffset: 0 })
        setState('done')
      })
    },
    { scope: root }
  )

  const togglePause = () => {
    const next = !paused
    setPaused(next)
    const tl = film.current
    if (tl && tl.progress() < 1) (next ? tl.pause() : tl.resume())
    if (ambient.current) (next ? ambient.current.pause() : ambient.current.resume())
  }

  const skip = () => film.current?.progress(1)

  const replay = () => {
    setState('film')
    setPaused(false)
    ambient.current?.kill()
    ambient.current = null
    window.scrollTo({ top: 0 })
    film.current?.restart()
  }

  return (
    <section
      ref={root}
      className={`film ${paused ? 'is-paused' : ''} is-${state}`}
      aria-label="SynCura introduction"
    >
      <div className="film-frame">
        {SCENES.map((scene, i) => (
          <canvas key={scene.id} className="film-layer" ref={(el) => (canvases.current[i] = el)} aria-hidden="true" />
        ))}

        <svg className="film-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <path id="film-arc" d={WORD_ARC} />
            <radialGradient id="final-halo" gradientUnits="userSpaceOnUse" cx="0" cy="0" r={R + 460}>
              <stop offset="0" stopColor="#020808" />
              <stop offset={R / (R + 460) - 0.004} stopColor="#041010" />
              <stop offset={R / (R + 460)} stopColor="#7cf2e2" />
              <stop offset={R / (R + 460) + 0.02} stopColor="#2fb7aa" />
              <stop offset={R / (R + 460) + 0.07} stopColor="#0f5454" />
              <stop offset={R / (R + 460) + 0.14} stopColor="#062222" />
              <stop offset="1" stopColor="#020808" stopOpacity="0" />
            </radialGradient>
          </defs>

          <g className="film-final">
            <rect className="final-bg" x="0" y="0" width={W} height={H} />
            <g className="final-world" transform={`translate(${CX} ${FINAL_Y}) scale(1)`}>
              <circle className="final-halo" r={R + 460} fill="url(#final-halo)" />
              <circle className="final-disc" r={R} />
              <circle className="final-rim" r={R} />
              <path className="final-ecg" d={rimPath} />
              <circle className="final-pen" r="7" />
            </g>
            <g transform={`translate(0 ${FINAL_DROP})`}>
              <text className="final-word" textAnchor="middle">
                <textPath href="#film-arc" startOffset="50%">SynCura</textPath>
              </text>
            </g>
          </g>

          <text className="film-word" textAnchor="middle">
            <textPath href="#film-arc" startOffset="50%">Every</textPath>
          </text>
        </svg>

        <div className="film-paper grid-paper" aria-hidden="true" />

        <svg className="ring-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <g className="ring">
            <circle className="ring-track" pathLength="100" />
            <circle className="ring-value" pathLength="100" strokeDasharray="100" strokeDashoffset="100" />
          </g>
        </svg>

        <div className="film-grain" aria-hidden="true" />
      </div>

      <div className="film-copy">
        <h1 className="film-title">
          <span>An early warning,</span> <span>read from the last</span> <span>90 minutes of vitals.</span>
        </h1>
        <p className="film-lede">
          SynCura scores each ICU patient’s risk of deterioration from the recent trace of their vitals. A research
          prototype built on PhysioNet ICU data.
        </p>
        <div className="film-actions">
          <Link to="/dashboard" className="btn film-btn-primary">Open the central station</Link>
          <a href="#how" className="btn film-btn-ghost">How a score is made</a>
        </div>
      </div>

      <div className="ring-stage">
        <div className="ring-copy">
          <h2>That horizon is a patient.</h2>
          <p>
            Pull back and each curve in the film is a risk ring. On the central station every bed gets one: a risk
            from 0 to 100, rescored every minute from the last 90 minutes of vitals.
          </p>
          <dl className="ring-vitals" aria-label="Example bed ICU-04 (synthetic)">
            <div style={{ '--c': 'var(--hr)' }}><dt>HR</dt><dd className="num">104</dd></div>
            <div style={{ '--c': 'var(--spo2)' }}><dt>SpO2</dt><dd className="num">93<small>%</small></dd></div>
            <div style={{ '--c': 'var(--rr)' }}><dt>RR</dt><dd className="num">24</dd></div>
            <div style={{ '--c': 'var(--temp)' }}><dt>Temp</dt><dd className="num">38.1<small>°C</small></dd></div>
          </dl>
          <Link to="/dashboard" className="btn btn-primary">See every bed</Link>
        </div>
        <div className="ring-slot">
          <div className="ring-num" role="img" aria-label={`Example: ICU-04, deterioration risk ${RISK} percent`}>
            <span className="num ring-num-value">{RISK}</span>
            <span className="ring-num-label">ICU-04 risk</span>
          </div>
        </div>
      </div>

      <div className="film-controls">
        <button type="button" className="film-control" onClick={togglePause} aria-pressed={paused}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            {paused ? <path d="M4 2.5v11l9-5.5z" /> : <path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" />}
          </svg>
          {paused ? 'Play' : 'Pause'}
        </button>
        {state === 'film' ? (
          <button type="button" className="film-control" onClick={skip}>Skip intro</button>
        ) : (
          <button type="button" className="film-control" onClick={replay}>Replay film</button>
        )}
        <span className="film-progress" aria-hidden="true"><span /></span>
      </div>
    </section>
  )
}
