// Shared website configuration. The displayed version is sourced from
// frontend/package.json so releases are tracked by changing that version.
import { version as APP_VERSION } from '../package.json'

// Training UI is internal-only. Public builds omit the nav tab and routes
// unless explicitly enabled at build time:
//
//   VITE_ENABLE_TRAINING=true npm run dev
export const TRAINING_ENABLED = import.meta.env.VITE_ENABLE_TRAINING === 'true'

export { APP_VERSION }
