import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import { APP_VERSION, TRAINING_ENABLED } from '../config'
import { useSimulation } from '../simulationContext'
import { BrandMark, Wordmark } from './Brand'

const NAV = [
  { to: '/dashboard', label: 'Central station' },
  { to: '/waveforms', label: 'Waveforms' },
  { to: '/simulated-data', label: 'Data feed' },
  { to: '/training', label: 'Training', internal: true },
  { to: '/architecture', label: 'Architecture' },
].filter((item) => TRAINING_ENABLED || !item.internal)

export function ThemeSwitch({ theme, onToggleTheme }) {
  const [side, setSide] = React.useState(theme)
  const [moving, setMoving] = React.useState(false)
  const timer = React.useRef(null)

  React.useEffect(() => () => window.clearTimeout(timer.current), [])

  const choose = (next) => {
    if (next === side) return
    setSide(next)
    setMoving(true)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      onToggleTheme()
      setMoving(false)
    }, 600)
  }

  return (
    <div className={`segmented theme-switch${moving ? ' is-moving' : ''}`} role="group" aria-label="Display mode" data-side={side}>
      <span className="theme-thumb" aria-hidden="true" />
      <button type="button" aria-pressed={side === 'light'} onClick={() => choose('light')}>
        Light
      </button>
      <button type="button" aria-pressed={side === 'dark'} onClick={() => choose('dark')}>
        Dark
      </button>
    </div>
  )
}

export function TopBar({ theme, onToggleTheme, landing = false, status }) {
  const { backendVersion } = useSimulation()
  const versionLabel =
    backendVersion && backendVersion !== APP_VERSION
      ? `Website version ${APP_VERSION}, backend reports ${backendVersion}`
      : `Website version ${APP_VERSION}`
  const versionText =
    backendVersion && backendVersion !== APP_VERSION ? `v${APP_VERSION} • API v${backendVersion}` : `v${APP_VERSION}`
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
        <span className="app-version num" aria-label={versionLabel}>
          {versionText}
        </span>
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
