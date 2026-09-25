// Procedural "material" frames for the hero film.
// Every frame shares one geometry: the top of a huge disc forms a horizon in
// the lower half of a 1920x1080 design space (like a macro lens on a curved
// surface). Painters draw in design units; canvases are 1440x810.

export const W = 1920
export const H = 1080
export const CX = W / 2
export const R = 1500
export const ARC_TOP = 640
export const CY = ARC_TOP + R
export const CANVAS_W = 1440
export const CANVAS_H = 810

// ECG beat template: phase 0..1 -> outward displacement (-1..1)
const BEAT = [[0, 0], [0.1, 0], [0.16, 0.1], [0.22, 0], [0.3, 0], [0.33, -0.14], [0.37, 1], [0.41, -0.32], [0.45, 0], [0.58, 0], [0.66, 0.22], [0.74, 0], [1, 0]]
export function ecg(phase) {
  const f = phase - Math.floor(phase)
  for (let i = 1; i < BEAT.length; i++) {
    const [x1, y1] = BEAT[i]
    if (f <= x1) {
      const [x0, y0] = BEAT[i - 1]
      return y0 + ((f - x0) / (x1 - x0)) * (y1 - y0)
    }
  }
  return 0
}

/** Points of an ECG running along a circle (centre cx,cy) at radius r. */
export function rimEcg({ cx = CX, cy = CY, r, amp = 60, beats = 11, span = 0.78, steps = 900 }) {
  const pts = []
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    const a = (t - 0.5) * span * 2 // radians from top
    const rr = r + ecg(t * beats) * amp
    pts.push([cx + Math.sin(a) * rr, cy - Math.cos(a) * rr])
  }
  return pts
}

function rng(seed) {
  let s = seed % 2147483647
  if (s <= 0) s += 2147483646
  return () => (s = (s * 16807) % 2147483647) / 2147483647
}

function disc(ctx, r = R) {
  ctx.beginPath()
  ctx.arc(CX, CY, r, 0, Math.PI * 2)
}

function withDisc(ctx, fn) {
  ctx.save()
  disc(ctx)
  ctx.clip()
  fn()
  ctx.restore()
}

function specks(ctx, rand, n, color, maxR = 1.6, y0 = 0) {
  ctx.fillStyle = color
  for (let i = 0; i < n; i++) {
    ctx.globalAlpha = rand() * 0.8
    ctx.beginPath()
    ctx.arc(rand() * W, y0 + rand() * (H - y0), rand() * maxR, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

function blotches(ctx, rand, n, color, rMin, rMax, alpha, y0 = 0) {
  for (let i = 0; i < n; i++) {
    const x = rand() * W
    const y = y0 + rand() * (H - y0)
    const r = rMin + rand() * (rMax - rMin)
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, color)
    g.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.globalAlpha = alpha * (0.4 + rand() * 0.6)
    ctx.fillStyle = g
    ctx.fillRect(x - r, y - r, r * 2, r * 2)
  }
  ctx.globalAlpha = 1
}

function polyline(ctx, pts) {
  ctx.beginPath()
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
}

function chalkLine(ctx, rand, x1, y1, x2, y2, width = 3) {
  for (let k = 0; k < 3; k++) {
    ctx.globalAlpha = 0.35 + rand() * 0.4
    ctx.lineWidth = width * (0.6 + rand() * 0.6)
    ctx.beginPath()
    const j = () => (rand() - 0.5) * 3
    ctx.moveTo(x1 + j(), y1 + j())
    const mx = (x1 + x2) / 2 + j() * 2
    const my = (y1 + y2) / 2 + j() * 2
    ctx.quadraticCurveTo(mx, my, x2 + j(), y2 + j())
    ctx.stroke()
  }
  ctx.globalAlpha = 1
}

/* ------------------------------------------------------------------ scenes */

function atmosphere(ctx) {
  const rand = rng(11)
  ctx.fillStyle = '#04070a'
  ctx.fillRect(0, 0, W, H)
  const g = ctx.createRadialGradient(CX, CY, R - 30, CX, CY, R + 420)
  g.addColorStop(0, '#06090c')
  g.addColorStop(0.055, '#0b0f12')
  g.addColorStop(0.07, '#ff7a3a')
  g.addColorStop(0.1, '#f3b25c')
  g.addColorStop(0.16, '#5aa6b8')
  g.addColorStop(0.32, '#1c4a63')
  g.addColorStop(0.62, '#0a1a28')
  g.addColorStop(1, '#04070a')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, W, H)
  withDisc(ctx, () => {
    const d = ctx.createLinearGradient(0, ARC_TOP, 0, H)
    d.addColorStop(0, '#0b0e11')
    d.addColorStop(1, '#030405')
    ctx.fillStyle = d
    ctx.fillRect(0, 0, W, H)
  })
  specks(ctx, rand, 160, '#cfe7ff', 1.2)
}

function ecgPaper(ctx) {
  const rand = rng(23)
  ctx.fillStyle = '#132228'
  ctx.fillRect(0, 0, W, H)
  specks(ctx, rand, 500, '#2c4650', 1.4)
  withDisc(ctx, () => {
    ctx.fillStyle = '#f8eee9'
    ctx.fillRect(0, 0, W, H)
    blotches(ctx, rand, 30, '#e9cfc6', 80, 260, 0.5, ARC_TOP)
    for (let x = 0; x <= W; x += 12) {
      ctx.strokeStyle = x % 60 === 0 ? 'rgba(206,74,70,0.55)' : 'rgba(206,74,70,0.22)'
      ctx.lineWidth = x % 60 === 0 ? 1.6 : 1
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    }
    for (let y = ARC_TOP - 60; y <= H; y += 12) {
      ctx.strokeStyle = y % 60 === 0 ? 'rgba(206,74,70,0.55)' : 'rgba(206,74,70,0.22)'
      ctx.lineWidth = y % 60 === 0 ? 1.6 : 1
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
    }
    ctx.strokeStyle = '#1b1f24'
    ctx.lineWidth = 4
    ctx.lineJoin = 'round'
    polyline(ctx, rimEcg({ r: R - 150, amp: 95, beats: 9 }))
    ctx.stroke()
  })
}

function bloodSmear(ctx) {
  const rand = rng(37)
  ctx.fillStyle = '#0a0406'
  ctx.fillRect(0, 0, W, H)
  withDisc(ctx, () => {
    const g = ctx.createLinearGradient(0, ARC_TOP, 0, H)
    g.addColorStop(0, '#c43a74')
    g.addColorStop(1, '#7c1a48')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, H)
    blotches(ctx, rand, 90, '#f08cb4', 20, 90, 0.35, ARC_TOP)
    blotches(ctx, rand, 60, '#5c0f33', 30, 110, 0.35, ARC_TOP)
    for (let i = 0; i < 70; i++) cell(ctx, rand() * W, ARC_TOP + 40 + rand() * 420, 14 + rand() * 16, 0.35)
  })
  // Red cells resting on the rim, like droplets on a lens
  for (let i = 0; i < 26; i++) {
    const a = (rand() - 0.5) * 1.4
    const r = 14 + rand() * 30
    const rr = R + r * 0.35
    cell(ctx, CX + Math.sin(a) * rr, CY - Math.cos(a) * rr, r, 1)
  }

  function cell(c, x, y, r, alpha) {
    const g = c.createRadialGradient(x - r * 0.2, y - r * 0.25, r * 0.1, x, y, r)
    g.addColorStop(0, '#f6a07e')
    g.addColorStop(0.45, '#e2523c')
    g.addColorStop(1, '#9b1f1c')
    c.globalAlpha = alpha
    c.fillStyle = g
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill()
    c.globalAlpha = alpha * 0.5
    c.fillStyle = '#ffd2bd'
    c.beginPath(); c.arc(x - r * 0.25, y - r * 0.3, r * 0.22, 0, Math.PI * 2); c.fill()
    c.globalAlpha = 1
  }
}

function blueprint(ctx) {
  const rand = rng(41)
  ctx.fillStyle = '#2b5c90'
  ctx.fillRect(0, 0, W, H)
  specks(ctx, rand, 2600, '#6f96c4', 1.3)
  specks(ctx, rand, 1200, '#1d4675', 1.5)
  ctx.strokeStyle = '#f4f7fb'
  ctx.lineCap = 'round'
  // Horizon drawn as a doubled chalk arc
  for (const off of [0, 22]) {
    const pts = rimEcg({ r: R - off, amp: 0, beats: 1, steps: 60 })
    for (let i = 1; i < pts.length; i++) chalkLine(ctx, rand, ...pts[i - 1], ...pts[i], 3)
  }
  // Ruler ticks along the rim
  for (let i = -22; i <= 22; i++) {
    const a = i * 0.03
    const r1 = R - 22
    const r2 = R - 22 - (i % 5 === 0 ? 70 : 34)
    chalkLine(ctx, rand, CX + Math.sin(a) * r1, CY - Math.cos(a) * r1, CX + Math.sin(a) * r2, CY - Math.cos(a) * r2, 2.4)
  }
  // Bronchial tree
  const branch = (x, y, ang, len, depth) => {
    if (depth === 0) return
    const x2 = x + Math.sin(ang) * len
    const y2 = y + Math.cos(ang) * len
    chalkLine(ctx, rand, x, y, x2, y2, 1.4 + depth * 0.7)
    const spread = 0.38 + rand() * 0.25
    branch(x2, y2, ang - spread, len * (0.68 + rand() * 0.1), depth - 1)
    branch(x2, y2, ang + spread, len * (0.68 + rand() * 0.1), depth - 1)
  }
  chalkLine(ctx, rand, CX, 60, CX, 300, 5)
  branch(CX, 300, -0.55, 180, 6)
  branch(CX, 300, 0.55, 180, 6)
  // Construction lines
  for (let i = 0; i < 5; i++) chalkLine(ctx, rand, 40 + i * 26, 40, 380 + i * 30, 360, 2)
  chalkLine(ctx, rand, 1500, 60, 1880, 340, 2)
  chalkLine(ctx, rand, 1560, 40, 1880, 250, 2)
}

function tissue(ctx) {
  const rand = rng(53)
  ctx.fillStyle = '#efe4ec'
  ctx.fillRect(0, 0, W, H)
  specks(ctx, rand, 300, '#d9c6d4', 1.4)
  withDisc(ctx, () => {
    ctx.fillStyle = '#f2cfe0'
    ctx.fillRect(0, 0, W, H)
    const step = 36
    for (let y = ARC_TOP - 20; y < H + step; y += step * 0.86) {
      const row = Math.round(y / step)
      for (let x = (row % 2) * (step / 2); x < W + step; x += step) {
        const cx = x + (rand() - 0.5) * 10
        const cy = y + (rand() - 0.5) * 10
        const r = 13 + rand() * 6
        ctx.fillStyle = rand() > 0.9 ? '#e6a8c8' : '#f7dcea'
        ctx.strokeStyle = '#c682ab'
        ctx.lineWidth = 2
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
        if (rand() > 0.35) {
          ctx.fillStyle = rand() > 0.5 ? '#5b2a86' : '#3f2a8f'
          ctx.globalAlpha = 0.85
          ctx.beginPath(); ctx.arc(cx + (rand() - 0.5) * 6, cy + (rand() - 0.5) * 6, 4 + rand() * 4, 0, Math.PI * 2); ctx.fill()
          ctx.globalAlpha = 1
        }
      }
    }
    // Vessels
    for (let i = 0; i < 7; i++) {
      const x = 150 + rand() * 1620
      const y = ARC_TOP + 110 + rand() * 300
      ctx.fillStyle = '#fff5fa'; ctx.strokeStyle = '#7b2e8e'; ctx.lineWidth = 5
      ctx.beginPath(); ctx.arc(x, y, 34 + rand() * 20, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
      ctx.fillStyle = '#3aa89c'; ctx.globalAlpha = 0.7
      ctx.beginPath(); ctx.arc(x + 4, y + 6, 12 + rand() * 8, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1
    }
  })
  ctx.strokeStyle = '#5a1f78'
  ctx.lineWidth = 9
  disc(ctx)
  ctx.stroke()
}

function xray(ctx) {
  const rand = rng(67)
  ctx.fillStyle = '#020304'
  ctx.fillRect(0, 0, W, H)
  withDisc(ctx, () => {
    const g = ctx.createLinearGradient(0, ARC_TOP, 0, H)
    g.addColorStop(0, '#2a3a4a')
    g.addColorStop(1, '#0c141c')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, H)
    ctx.lineCap = 'round'
    // Ribs: soft glowing arcs
    for (let i = 0; i < 7; i++) {
      const y = ARC_TOP + 40 + i * 62
      for (const side of [-1, 1]) {
        for (const [w, a] of [[34, 0.07], [18, 0.16], [7, 0.45]]) {
          ctx.strokeStyle = `rgba(214,232,245,${a})`
          ctx.lineWidth = w
          ctx.beginPath()
          ctx.moveTo(CX + side * 70, y)
          ctx.bezierCurveTo(CX + side * 420, y - 70, CX + side * 820, y + 10, CX + side * 980, y + 150)
          ctx.stroke()
        }
      }
    }
    // Spine
    for (let y = ARC_TOP - 10; y < H; y += 46) {
      ctx.fillStyle = 'rgba(214,232,245,0.35)'
      ctx.fillRect(CX - 44, y, 88, 36)
    }
    blotches(ctx, rand, 40, '#8fb3cf', 60, 200, 0.12, ARC_TOP)
  })
  ctx.strokeStyle = 'rgba(214,232,245,0.8)'
  ctx.lineWidth = 3
  disc(ctx)
  ctx.stroke()
}

function pulseGlow(ctx) {
  const rand = rng(79)
  ctx.fillStyle = '#0d0203'
  ctx.fillRect(0, 0, W, H)
  const halo = ctx.createRadialGradient(CX, ARC_TOP + 200, 50, CX, ARC_TOP + 200, 900)
  halo.addColorStop(0, 'rgba(255,60,40,0.55)')
  halo.addColorStop(1, 'rgba(255,60,40,0)')
  ctx.fillStyle = halo
  ctx.fillRect(0, 0, W, H)
  withDisc(ctx, () => {
    const g = ctx.createRadialGradient(CX, ARC_TOP + 120, 20, CX, ARC_TOP + 160, 1100)
    g.addColorStop(0, '#ffb08a')
    g.addColorStop(0.18, '#ff4a33')
    g.addColorStop(0.5, '#a3121a')
    g.addColorStop(1, '#2a0306')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, H)
    specks(ctx, rand, 2200, 'rgba(90,6,10,0.8)', 2.2, ARC_TOP)
    // Skin ridges
    ctx.strokeStyle = 'rgba(255,190,160,0.12)'
    ctx.lineWidth = 3
    for (let k = 1; k < 16; k++) {
      polyline(ctx, rimEcg({ r: R - k * 26, amp: 0, beats: 1, steps: 80 }))
      ctx.stroke()
    }
  })
  ctx.shadowColor = '#ff6a4a'
  ctx.shadowBlur = 30
  ctx.strokeStyle = '#ff9a7a'
  ctx.lineWidth = 4
  disc(ctx)
  ctx.stroke()
  ctx.shadowBlur = 0
}

function monitor(ctx) {
  const rand = rng(83)
  ctx.fillStyle = '#010807'
  ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = 'rgba(61,240,142,0.07)'
  for (let x = 20; x < W; x += 40) for (let y = 20; y < H; y += 40) ctx.fillRect(x, y, 2, 2)
  withDisc(ctx, () => {
    ctx.fillStyle = '#021410'
    ctx.fillRect(0, 0, W, H)
    blotches(ctx, rand, 20, '#0b3a2c', 80, 300, 0.5, ARC_TOP)
  })
  ctx.lineJoin = 'round'
  for (const [w, a, blur] of [[10, 0.25, 30], [3.5, 1, 12]]) {
    ctx.shadowColor = '#3df08e'
    ctx.shadowBlur = blur
    ctx.strokeStyle = `rgba(61,240,142,${a})`
    ctx.lineWidth = w
    polyline(ctx, rimEcg({ r: R + 30, amp: 110, beats: 10 }))
    ctx.stroke()
  }
  ctx.shadowBlur = 0
}

function sketch(ctx) {
  const rand = rng(97)
  ctx.fillStyle = '#e8dcc0'
  ctx.fillRect(0, 0, W, H)
  blotches(ctx, rand, 26, '#d4c19b', 60, 260, 0.22)
  specks(ctx, rand, 2400, '#a8926a', 1.1)
  ctx.strokeStyle = '#2a211a'
  ctx.lineCap = 'round'
  // Scratchy double outline
  for (let k = 0; k < 3; k++) {
    ctx.globalAlpha = 0.5 + rand() * 0.4
    ctx.lineWidth = 1.5 + rand() * 1.5
    polyline(ctx, rimEcg({ r: R - k * 6 + (rand() - 0.5) * 8, amp: 0, beats: 1, steps: 90 }))
    ctx.stroke()
  }
  // Vessels growing from the rim
  const vessel = (x, y, ang, len, depth) => {
    if (depth === 0 || y > H + 20) return
    ctx.globalAlpha = 0.55 + rand() * 0.4
    ctx.lineWidth = 0.8 + depth * 0.55
    ctx.beginPath()
    ctx.moveTo(x, y)
    let px = x
    let py = y
    for (let s = 0; s < 6; s++) {
      ang += (rand() - 0.5) * 0.5
      px += Math.sin(ang) * (len / 6)
      py += Math.cos(ang) * (len / 6)
      ctx.lineTo(px, py)
    }
    ctx.stroke()
    if (rand() > 0.25) vessel(px, py, ang - 0.4 - rand() * 0.4, len * 0.72, depth - 1)
    if (rand() > 0.25) vessel(px, py, ang + 0.4 + rand() * 0.4, len * 0.72, depth - 1)
  }
  for (let i = 0; i < 14; i++) {
    const a = (rand() - 0.5) * 1.3
    vessel(CX + Math.sin(a) * R, CY - Math.cos(a) * R, a * 0.3, 90 + rand() * 90, 4)
  }
  // Cross-hatching
  ctx.globalAlpha = 0.35
  ctx.lineWidth = 1
  for (let i = 0; i < 140; i++) {
    const x = 300 + rand() * 1320
    const y = ARC_TOP + 60 + rand() * 360
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 26, y - 26); ctx.stroke()
  }
  ctx.globalAlpha = 1
}

// Order, duration (s) and the word riding the horizon for each cut.
export const SCENES = [
  { id: 'atmosphere', paint: atmosphere, word: 'Every', ink: '#e9eef2', dur: 1.25 },
  { id: 'ecg-paper', paint: ecgPaper, word: 'Every', ink: '#f8eee9', dur: 0.9 },
  { id: 'blood', paint: bloodSmear, word: 'minute', ink: '#f7dfe8', dur: 0.85 },
  { id: 'blueprint', paint: blueprint, word: 'minute', ink: '#f4f7fb', dur: 0.8 },
  { id: 'tissue', paint: tissue, word: 'leaves a trace.', ink: '#3b1a52', dur: 0.8 },
  { id: 'xray', paint: xray, word: 'leaves a trace.', ink: '#e5eef6', dur: 0.66 },
  { id: 'pulse', paint: pulseGlow, word: 'leaves a trace.', ink: '#ffe6dc', dur: 0.66 },
  { id: 'monitor', paint: monitor, word: 'leaves a trace.', ink: '#d9ffe9', dur: 0.66 },
  { id: 'sketch', paint: sketch, word: 'leaves a trace.', ink: '#2a211a', dur: 0.95 },
]

export function paintScene(canvas, scene) {
  canvas.width = CANVAS_W
  canvas.height = CANVAS_H
  const ctx = canvas.getContext('2d')
  ctx.setTransform(CANVAS_W / W, 0, 0, CANVAS_H / H, 0, 0)
  scene.paint(ctx)
}
