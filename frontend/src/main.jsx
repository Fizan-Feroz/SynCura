import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import '@fontsource-variable/archivo/wdth.css'
import './theme/tokens.css'
import './styles/base.css'
import './styles/layout.css'
import './styles/landing.css'
import './styles/film.css'
import './styles/station.css'
import './styles/pages.css'

createRoot(document.getElementById('root')).render(<App />)
