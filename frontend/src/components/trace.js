// ECG-style trace generation (Lead II shape: P wave, QRS complex, T wave).

function beat(x0, w, base, amp) {
  // Points for one beat starting at x0, width w, baseline y `base`.
  const at = (f) => x0 + w * f
  return [
    [at(0.1), base],
    [at(0.16), base - amp * 0.1],   // P
    [at(0.22), base],
    [at(0.3), base],
    [at(0.33), base + amp * 0.14],  // Q
    [at(0.37), base - amp],         // R
    [at(0.41), base + amp * 0.32],  // S
    [at(0.45), base],
    [at(0.58), base],
    [at(0.66), base - amp * 0.22],  // T
    [at(0.74), base],
    [at(1), base],
  ]
}

/** SVG path of `beats` heartbeats across [x0, x1] around baseline `base`. */
export function ecgPath({ x0 = 0, x1 = 1000, base = 100, amp = 70, beats = 8, jitter = 0.08, seed = 7 }) {
  let s = seed
  const rand = () => {
    s = (s * 16807) % 2147483647
    return s / 2147483647
  }
  const w = (x1 - x0) / beats
  const pts = [[x0, base]]
  for (let i = 0; i < beats; i++) {
    const a = amp * (1 - jitter / 2 + rand() * jitter)
    pts.push(...beat(x0 + i * w, w, base, a))
  }
  return pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')
}

/** Smooth path through data values scaled into a width x height box. */
export function seriesPath(values, width, height, pad = 4, domain) {
  if (!values.length) return ''
  const min = domain ? domain[0] : Math.min(...values)
  const max = domain ? domain[1] : Math.max(...values)
  const range = max - min || 1
  const step = (width - pad * 2) / Math.max(1, values.length - 1)
  const pts = values.map((v, i) => [pad + i * step, pad + (height - pad * 2) * (1 - (v - min) / range)])
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    const [px, py] = pts[i - 1]
    const [x, y] = pts[i]
    const cx = (px + x) / 2
    d += ` C${cx.toFixed(1)} ${py.toFixed(1)} ${cx.toFixed(1)} ${y.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
}

export function riskTone(risk) {
  if (risk >= 85) return 'critical'
  if (risk >= 70) return 'high'
  if (risk >= 45) return 'watch'
  return 'stable'
}

export function riskLabel(risk) {
  return { critical: 'Critical', high: 'High', watch: 'Watch', stable: 'Stable' }[riskTone(risk)]
}
