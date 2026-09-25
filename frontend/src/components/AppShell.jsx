import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import { BrandMark, Wordmark } from './Brand'

const NAV = [
  { to: '/dashboard', label: 'Central station' },
  { to: '/waveforms', label: 'Waveforms' },
  { to: '/simulated-data', label: 'Data feed' },
  { to: '/training', label: 'Training' },
  { to: '/architecture', label: 'Architecture' },
]

export function ThemeSwitch({ theme, onToggleTheme }) {
  return (
    <div className="segmented theme-switch" role="group" aria-label="Display mode">
      <button type="button" aria-pressed={theme === 'light'} onClick={() => theme !== 'light' && onToggleTheme()}>
        Paper
      </button>
      <button type="button" aria-pressed={theme === 'dark'} onClick={() => theme !== 'dark' && onToggleTheme()}>
        Monitor
      </button>
    </div>
  )
}

export function TopBar({ theme, onToggleTheme, landing = false, status }) {
  return (
    <header className={`topbar ${landing ? 'topbar-landing' : ''}`}>
      <Link to="/" className="topbar-brand" aria-label="SynCura home">
        <BrandMark />
        <Wordmark />
      </Link>

      <nav className="topbar-nav" aria-label="Primary">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `topbar-link ${isActive ? 'is-active' : ''}`}
            end={item.to === '/training' ? false : undefined}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="topbar-end">
        {status && (
          <span className={`live-status ${status.live ? 'is-live' : ''}`} aria-live="polite">
            <span className="live-dot" aria-hidden="true" />
            {status.label}
          </span>
        )}
        <ThemeSwitch theme={theme} onToggleTheme={onToggleTheme} />
      </div>
    </header>
  )
}

export default function AppShell({ theme, onToggleTheme, status, children, wide = false }) {
  return (
    <div className="app">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <TopBar theme={theme} onToggleTheme={onToggleTheme} status={status} />
      <main className={`app-main ${wide ? 'app-main-wide' : ''}`} id="main-content" tabIndex="-1">
        {children}
      </main>
    </div>
  )
}
