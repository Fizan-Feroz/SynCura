import React, { useEffect, useState } from 'react'
import { getSoundEnabled, setSoundEnabled } from '../utils/sounds'

export function SoundToggle({ className = '' }) {
  const [enabled, setEnabled] = useState(() => getSoundEnabled())

  useEffect(() => {
    const stored = getSoundEnabled()
    setEnabled(stored)
  }, [])

  const toggle = () => {
    const next = !enabled
    setEnabled(next)
    setSoundEnabled(next)
  }

  return (
    <button
      type="button"
      className={`btn btn-quiet btn-sm sound-toggle ${className}`}
      onClick={toggle}
      aria-pressed={enabled}
      aria-label={enabled ? 'Sound alerts on' : 'Sound alerts off'}
      title={enabled ? 'Sound alerts: on' : 'Sound alerts: off'}
    >
      <span aria-hidden="true">
        {enabled ? '🔊' : '🔇'}
      </span>
      <span className="sound-toggle-label">{enabled ? 'Sound on' : 'Sound off'}</span>
    </button>
  )
}