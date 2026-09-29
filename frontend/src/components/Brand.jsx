import React from 'react'
import markLight from '../assets/syncura-mark-light.svg'
import markDark from '../assets/syncura-mark-dark.svg'

// The SynCura logo mark (heart + circuit). Light ink for paper mode, bright
// ink for monitor mode — picked by the active theme, not the OS setting.
export function BrandMark({ theme = 'light', title = 'SynCura logo' }) {
  return (
    <img
      className="brand-mark"
      src={theme === 'dark' ? markDark : markLight}
      alt={title}
    />
  )
}

export function Wordmark() {
  return (
    <span className="wordmark">
      Syn<span>Cura</span>
    </span>
  )
}
