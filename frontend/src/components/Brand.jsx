import React from 'react'

// The SynCura mark (heart, trace, model nodes) redrawn as strokes so it
// follows the current ink colour and can be drawn on by GSAP.
export function BrandMark({ className = '', title }) {
  return (
    <svg
      className={`brand-mark ${className}`.trim()}
      viewBox="0 0 64 64"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <path className="mark-heart" d="M32 56 C18 46 6 36 6 22 C6 13 12.5 8 19.5 8 C25.5 8 29.5 11.5 32 16 C34.5 11.5 38.5 8 44.5 8 C51.5 8 58 13 58 22 C58 29 55 35 51 40" />
      <path className="mark-trace" d="M9 32 H15.5 L18 27 L21.5 41 L25.5 18.5 L31 33 L37 18.5 L40.5 33" />
      <path className="mark-nodes" d="M40.5 28.5 H44.5 L49 23.5 M40.5 32.5 H52 M40.5 35.5 H44.5 L49 41" />
      <circle className="mark-node" cx="51.5" cy="21" r="2.4" />
      <circle className="mark-node" cx="54.5" cy="32.5" r="2.4" />
      <circle className="mark-node" cx="51.5" cy="43.5" r="2.4" />
    </svg>
  )
}

export function Wordmark() {
  return (
    <span className="wordmark">
      Syn<span>Cura</span>
    </span>
  )
}
