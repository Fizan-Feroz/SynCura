// Sound utility for alert cues. Uses Web Audio API for critical sound
// (generated programmatically) and a WAV file for warning.

let warningAudio = null
let criticalAudio = null
let audioContext = null
let soundEnabled = true

function getAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)()
  }
  return audioContext
}

export function initSounds() {
  if (warningAudio) return
  warningAudio = new Audio('/notification.wav')
  warningAudio.preload = 'auto'
  warningAudio.volume = 0.6
}

function playSound(audio) {
  if (!soundEnabled || !audio) return
  const ctx = getAudioContext()
  if (ctx.state === 'suspended') ctx.resume()
  audio.currentTime = 0
  audio.play().catch(() => { /* user interaction required */ })
}

function generateCriticalSound() {
  // Create a more urgent sound: two-tone siren-like pattern
  const ctx = getAudioContext()
  const duration = 0.4
  const sampleRate = ctx.sampleRate
  const samples = Math.ceil(sampleRate * duration)
  const buffer = ctx.createBuffer(1, samples, sampleRate)
  const data = buffer.getChannelData(0)

  // Two alternating tones: high (880Hz) and higher (1320Hz)
  const freq1 = 880
  const freq2 = 1320
  const switchInterval = 0.08 // switch every 80ms
  let currentFreq = freq1
  let nextSwitch = switchInterval

  for (let i = 0; i < samples; i++) {
    const t = i / sampleRate
    if (t >= nextSwitch) {
      currentFreq = currentFreq === freq1 ? freq2 : freq1
      nextSwitch += switchInterval
    }
    // Add envelope for less harsh attack
    const env = Math.min(1, t / 0.02) * Math.max(0, 1 - t / duration)
    data[i] = Math.sin(2 * Math.PI * currentFreq * t) * env * 0.3
  }

  const source = ctx.createBufferSource()
  source.buffer = buffer
  const gain = ctx.createGain()
  gain.gain.value = 0.5
  source.connect(gain).connect(ctx.destination)
  return { play: () => source.start() }
}

export function playWarning() {
  if (!soundEnabled) return
  initSounds()
  playSound(warningAudio)
}

export function playCritical() {
  if (!soundEnabled) return
  const ctx = getAudioContext()
  if (ctx.state === 'suspended') ctx.resume()
  const sound = generateCriticalSound()
  sound.play()
}

export function setSoundEnabled(enabled) {
  soundEnabled = enabled
  try { localStorage.setItem('syncura-sound-enabled', String(enabled)) } catch {}
}

export function getSoundEnabled() {
  try {
    const stored = localStorage.getItem('syncura-sound-enabled')
    if (stored !== null) return stored === 'true'
  } catch {}
  return true
}

// Initialize from storage on load
if (typeof window !== 'undefined') {
  soundEnabled = getSoundEnabled()
}