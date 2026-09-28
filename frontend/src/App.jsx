import React, { Suspense, lazy, useEffect, useState } from 'react'
import { BrowserRouter, Navigate, Routes, Route, useLocation } from 'react-router-dom'
import { TRAINING_ENABLED } from './config'
import { SimulationProvider } from './simulationContext'
import AppShell from './components/AppShell'

const LandingPage = lazy(() => import('./components/LandingPage'))
const Dashboard = lazy(() => import('./components/Dashboard'))
const TrainingConfig = lazy(() => import('./components/TrainingConfig'))
const TrainingMonitor = lazy(() => import('./components/TrainingMonitor'))
const TrainingJobsList = lazy(() => import('./components/TrainingJobsList'))
const SimulatedDataFeed = lazy(() => import('./components/SimulatedDataFeed'))
const ArchitecturePage = lazy(() => import('./components/ArchitecturePage'))
const BedProfile = lazy(() => import('./components/BedProfile'))
const SensorWaveform = lazy(() => import('./components/SensorWaveform'))
const AdminPanel = lazy(() => import('./components/AdminPanel'))

function initialTheme() {
  try {
    const stored = localStorage.getItem('syncura-theme')
    if (stored === 'dark' || stored === 'light') return stored
  } catch {
    /* storage unavailable */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function ScrollToTop() {
  const { pathname, hash } = useLocation()
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0)
  }, [pathname, hash])
  return null
}

function Loading() {
  return (
    <div className="page-loading" role="status">
      <span className="sr-only">Loading</span>
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme)
  const toggleTheme = () => {
    const apply = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'))
    if (document.startViewTransition) document.startViewTransition(apply)
    else apply()
  }

  useEffect(() => {
    document.documentElement.dataset.theme = theme === 'dark' ? 'monitor' : 'paper'
    try {
      localStorage.setItem('syncura-theme', theme)
    } catch {
      /* storage unavailable */
    }
  }, [theme])

  const shell = (children, props = {}) => (
    <AppShell theme={theme} onToggleTheme={toggleTheme} {...props}>{children}</AppShell>
  )

  return (
    <SimulationProvider>
      <BrowserRouter>
        <ScrollToTop />
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<LandingPage theme={theme} onToggleTheme={toggleTheme} />} />
            <Route path="/dashboard" element={<Dashboard theme={theme} onToggleTheme={toggleTheme} />} />
            <Route path="/bed/:patientId" element={<BedProfile theme={theme} onToggleTheme={toggleTheme} />} />
            <Route path="/simulated-data" element={shell(<SimulatedDataFeed />, { wide: true })} />
            <Route path="/waveforms" element={shell(<SensorWaveform />, { wide: true })} />
            {TRAINING_ENABLED ? (
              <>
                <Route path="/training" element={shell(<TrainingJobsList />)} />
                <Route path="/training/new" element={shell(<TrainingConfig />)} />
                <Route path="/training/:jobId" element={shell(<TrainingMonitor />)} />
              </>
            ) : (
              <>
                <Route path="/training" element={<Navigate to="/" replace />} />
                <Route path="/training/new" element={<Navigate to="/" replace />} />
                <Route path="/training/:jobId" element={<Navigate to="/" replace />} />
              </>
            )}
            <Route path="/architecture" element={shell(<ArchitecturePage />)} />
            <Route path="/admin" element={shell(<AdminPanel />, { wide: true })} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </SimulationProvider>
  )
}
