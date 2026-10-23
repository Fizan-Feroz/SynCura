// Alert cues. Both cues are plain HTMLAudioElement playback, which is the
// path browsers reliably allow after a user gesture. The critical siren is
// synthesised once into a WAV blob at runtime (no binary asset, no licence).

const STORAGE_KEY = 'syncura-sound-enabled'
const CRITICAL_SECONDS = 5

let warningAudio = null
let criticalAudio = null
let criticalStopTimer = null
let soundEnabled = true

function readStoredEnabled() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored !== null) return stored === 'true'
  } catch {
    /* private mode */
  }
  return true
}

if (typeof window !== 'undefined') {
  soundEnabled = readStoredEnabled()
}

/* ------------------------------------------------------------------ */
/* Critical siren synthesis                                            */
/* ------------------------------------------------------------------ */

function encodeWav(samples, sampleRate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(bytes)
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  ascii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }
  return bytes
}

// One 1s cycle of a two-tone ICU siren: 1000Hz / 760Hz alternating every
// 125ms, with a short fade on each step so it does not click.
function buildCriticalWavUrl() {
  const sampleRate = 22050
  const duration = 1
  const total = Math.floor(sampleRate * duration)
  const samples = new Float32Array(total)
  const hi = 1000
  const lo = 760
  const step = 0.125
  const fade = 0.012

  for (let i = 0; i < total; i += 1) {
    const t = i / sampleRate
    const segment = Math.floor(t / step)
    const freq = segment % 2 === 0 ? hi : lo
    const inSegment = t - segment * step
    const envelope =
      Math.min(1, inSegment / fade) * Math.min(1, (step - inSegment) / fade)
    // Fundamental plus a quiet octave keeps it audible on laptop speakers
    const tone =
      Math.sin(2 * Math.PI * freq * t) * 0.72 +
      Math.sin(2 * Math.PI * freq * 2 * t) * 0.16
    samples[i] = tone * envelope * 0.85
  }

  const bytes = encodeWav(samples, sampleRate)
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
}

function getWarningAudio() {
  if (!warningAudio) {
    warningAudio = new Audio('/notification.wav')
    warningAudio.preload = 'auto'
    warningAudio.volume = 0.7
  }
  return warningAudio
}

function getCriticalAudio() {
  if (!criticalAudio) {
    criticalAudio = new Audio(buildCriticalWavUrl())
    criticalAudio.preload = 'auto'
    criticalAudio.loop = true
    criticalAudio.volume = 0.9
  }
  return criticalAudio
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

export function initSounds() {
  getWarningAudio()
  getCriticalAudio()
}

// Called from a click handler so the browser unlocks both elements.
export function primeAudio() {
  initSounds()
  const critical = getCriticalAudio()
  const previousVolume = critical.volume
  critical.volume = 0
  critical.play().then(() => {
    critical.pause()
    critical.currentTime = 0
    critical.volume = previousVolume
  }).catch(() => {
    critical.volume = previousVolume
  })
}

// Browsers only allow audio after a gesture. Prime both elements on the
// first interaction anywhere on the page so cues can fire later on their own.
let unlockInstalled = false

export function installAudioUnlock() {
  if (unlockInstalled || typeof window === 'undefined') return
  unlockInstalled = true
  const unlock = () => {
    primeAudio()
    window.removeEventListener('pointerdown', unlock)
    window.removeEventListener('keydown', unlock)
    window.removeEventListener('touchstart', unlock)
  }
  window.addEventListener('pointerdown', unlock)
  window.addEventListener('keydown', unlock)
  window.addEventListener('touchstart', unlock)
}

export function playWarning() {
  if (!soundEnabled) return
  const audio = getWarningAudio()
  try {
    audio.currentTime = 0
    audio.play().catch(() => {})
  } catch {
    /* ignore */
  }
}

export function playCritical() {
  if (!soundEnabled) return
  const audio = getCriticalAudio()
  stopCritical()
  try {
    audio.currentTime = 0
    audio.play().catch(() => {})
  } catch {
    return
  }
  criticalStopTimer = setTimeout(() => {
    criticalStopTimer = null
    try {
      audio.pause()
      audio.currentTime = 0
    } catch {
      /* ignore */
    }
  }, CRITICAL_SECONDS * 1000)
}

export function stopCritical() {
  if (criticalStopTimer) {
    clearTimeout(criticalStopTimer)
    criticalStopTimer = null
  }
  if (criticalAudio) {
    try {
      criticalAudio.pause()
      criticalAudio.currentTime = 0
    } catch {
      /* ignore */
    }
  }
}

export function setSoundEnabled(enabled) {
  soundEnabled = enabled
  try {
    localStorage.setItem(STORAGE_KEY, String(enabled))
  } catch {
    /* private mode */
  }
  if (!enabled) stopCritical()
}

export function getSoundEnabled() {
  return soundEnabled
}

export { readStoredEnabled as getStoredSoundEnabled }
