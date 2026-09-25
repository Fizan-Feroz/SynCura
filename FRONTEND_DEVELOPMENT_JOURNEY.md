# SynCura Frontend Development Journey

## Purpose

This document records the evolution of the SynCura frontend from an initial research prototype into a coherent React application with a branded interface, simulated clinical workflows, training-job controls, accessible interaction patterns, and explicit research-prototype safeguards.

It is intended to preserve:

- the development context behind major interface decisions;
- the distinction between simulated and backend-derived information;
- the design and engineering practices that should guide future work;
- known limitations and a prioritized path toward a production-quality frontend.

The journey described here covers the first frontend implementation in May 2026 through the audit-driven theme and accessibility work completed in September 2026.

## Executive Summary

SynCura's frontend began as a functional way to demonstrate the machine-learning workflow. It first exposed training-job controls, then expanded into a clinical dashboard with patient ranking, alerts, threshold tuning, NEWS2 comparison, and explainability previews. The interface was then given a distinct SynCura identity, dark-mode support, a welcome experience, and an architecture page.

The latest phase moved beyond visual polish. A dedicated theme system, semantic color tokens, motion utilities, page-visibility handling, accessibility improvements, dependency updates, and research-prototype disclaimers were introduced. The result is more coherent and trustworthy, but the frontend is still a demonstration system rather than a clinical product.

The central engineering lesson is that the interface must continue to distinguish three different information classes:

1. synthetic data generated in the browser;
2. model metrics produced by offline evaluation;
3. live data that has actually passed through the backend inference pipeline.

Keeping those boundaries visible is more important than making the interface appear more production-like.

## Starting Point

The initial frontend was created alongside the backend and machine-learning pipeline in commit `f892f15`. Its primary purpose was functional: demonstrate that the SynCura model and backend could be represented through an interactive browser interface.

At that stage, the important product questions were:

- How should a clinician or reviewer scan a ranked patient queue?
- How should risk and threshold changes be communicated?
- How should model output and explainability be presented without overstating their meaning?
- How should model training be controlled and monitored from the same application?
- How should the system communicate that its displayed patient data is simulated?

These questions shaped the later information architecture and visual redesign.

## Development Timeline

| Date | Stage | Main outcome |
|---|---|---|
| 8 May 2026 | Project scaffold | React/Vite application, shared application shell, simulation context, and initial routes |
| 8 May 2026 | Training interface | Training configuration, job list, and job-monitoring views added |
| 8 May 2026 | Clinical dashboard | Ranked patient risk, alerts, threshold tuning, NEWS2 comparison, and impact previews added |
| 8 May 2026 | Simulation helpers | Synthetic patient fixtures and reusable scenario data introduced |
| 8-9 May 2026 | Brand and visual redesign | SynCura identity, welcome page, logo, responsive dashboard styling, and dark mode added |
| 9 May 2026 | Architecture and operations | Architecture page, startup guidance, and alert behavior expanded |
| 10-20 September 2026 | Research integration | Attention-LSTM, SHAP, causal-window, honesty, secret-handling, and audit improvements propagated into the product narrative |
| 24 September 2026 | Welcome-page polish | Hero presentation and sample dashboard preview refined |
| 25 September 2026 | Audit-driven theme redesign | Semantic tokens, motion controls, accessibility, performance behavior, disclaimers, and dependency maintenance introduced |
| 25 September 2026 | Current frontend audit | Remaining architecture, reliability, testing, and maintainability gaps documented for the next phase |

## Journey by Development Phase

### 1. Establishing a Functional Prototype

The first major addition was the machine-learning training interface, represented by commits `516e865` and `e10ba53`. It created a direct browser workflow for configuring training, listing jobs, and monitoring a selected job.

This phase established several conventions that still shape the application:

- React Router provides page-level navigation.
- Training views communicate with FastAPI through REST endpoints.
- Short polling intervals provide near-real-time progress without introducing a streaming client.
- Status is represented through badges, progress bars, metric cards, and error states.
- The frontend configuration is intended to mirror backend training parameters.

This was an important step because it changed SynCura from a collection of scripts into an operable research workflow.

### 2. Building the Clinical Dashboard

The dashboard phase added the main patient-monitoring experience. It includes:

- a ranked patient list;
- synthetic vital signs and risk trends;
- status and severity indicators;
- risk dials and sparklines;
- live alert cards;
- a threshold-tuning control;
- sensitivity, specificity, precision, and false-alarm calculations;
- a NEWS2-style comparison;
- per-patient synthetic impact previews.

The dashboard was designed around a queue-first workflow: reviewers first identify who needs attention, then inspect a selected patient's trends and supporting signals.

The initial implementation also established an important product convention: simulation controls belong near the data being presented. Users must be able to pause, reset, and change the scenario without confusing generated data for bedside telemetry.

### 3. Introducing Scenarios and Synthetic Data

Synthetic-data support evolved from static fixtures into a shared simulation context. The application now defines multiple scenarios, including:

- Baseline Mix;
- Respiratory Decline;
- Septic Shock;
- Cardiac Stress;
- Recovery Trend.

Patients are assigned trajectories such as severe, recovery, stable, or volatile. Each update derives new synthetic vitals, a risk value, a status label, and a short waveform.

This made the frontend useful for demonstrations and team walkthroughs without requiring continuous backend or dataset availability. It also created the most important data-boundary risk in the project: simulated and real inference results can easily be presented in the same visual language.

The current UI mitigates this through visible simulation banners and synthetic labels. Future work should strengthen that separation structurally rather than relying only on copy.

### 4. Establishing the SynCura Identity

Commit `cd930e0` introduced the branded welcome page, logo, and dashboard redesign. Later commits refined the hero layout, corrected the logo, improved dark mode, and added the architecture view.

This phase changed the application from a collection of working screens into a product-like prototype. The frontend gained:

- a landing experience;
- a consistent SynCura visual identity;
- a reusable application sidebar;
- route-specific pages;
- a clearer architecture narrative;
- light and dark themes.

The architecture page also helped communicate the relationship between ingestion, preprocessing, inference, risk scoring, and notifications. However, some language in that page has since become more ambitious than the current implementation supports, so future revisions should keep deployment, hardware, security, and scalability claims aligned with verified project behavior.

### 5. Adding Motion Responsibly

The current motion layer is deliberately small. It contains route transitions, section entrances, waveform-related animation support, and a live pulse. It also includes page-visibility state so simulation and presentation activity can pause when the page is hidden.

Commit `fed3485` introduced this structure, and commit `0684b64` corrected animation behavior so the experience respects `prefers-reduced-motion`. This is a good example of refining implementation after accessibility review rather than treating accessibility as a final checklist item.

The next phase should preserve the same restraint and avoid broad property transitions such as `transition-all`.

### 6. Hardening Trust and Accessibility

The audit-driven redesign improved the interface in four areas:

- semantic theme tokens for light and dark modes;
- visible keyboard focus treatment;
- a skip link and landmark structure;
- ARIA labels, live regions, progress semantics, and keyboard-operable controls;
- reduced-motion support;
- page-visibility-aware updates;
- research-prototype and synthetic-data disclaimers;
- clearer separation between model metrics and simulated dashboard output.

These changes are significant because clinical software should be understandable, keyboard accessible, and explicit about uncertainty. The application is still missing automated accessibility tests and a complete responsive review, but its foundational patterns are now stronger.

### 7. Current Audit and Reframing

The current audit found that the frontend's basic structure is sound, but several categories of work remain before it can be called production-ready.

The immediate priorities are reliability, data-source clarity, automated tests, and stricter frontend/backend contracts. The longer-term priorities are type safety, accessibility regression testing, production deployment configuration, and a deliberate decision about whether live backend data belongs inside the same dashboard as simulation mode.

## Current Frontend Architecture

### Application Layers

```text
frontend/
├── index.html               (carries a pre-paint theme script)
├── package.json
├── postcss.config.js        (Autoprefixer only)
├── vercel.json
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── api.js
    ├── simulationContext.jsx
    ├── theme/
    │   └── tokens.css
    ├── motion/
    │   └── gsap.js
    ├── styles/
    │   ├── base.css
    │   ├── layout.css
    │   ├── landing.css
    │   ├── film.css
    │   ├── station.css
    │   └── pages.css
    └── components/
        ├── AppShell.jsx
        ├── Brand.jsx
        ├── LandingPage.jsx
        ├── Dashboard.jsx
        ├── trace.js
        ├── ArchitecturePage.jsx
        ├── SimulatedDataFeed.jsx
        ├── SensorWaveform.jsx
        ├── TrainingConfig.jsx
        ├── TrainingJobsList.jsx
        ├── TrainingMonitor.jsx
        └── film/
            ├── HeroFilm.jsx
            └── scenes.js
```

### Responsibilities

- `main.jsx` loads the self-hosted variable font, then the theme tokens, then the
  stylesheets in a fixed order. `theme/tokens.css` must stay first — every later
  sheet consumes its custom properties.
- `App.jsx` owns routing (all routes are `React.lazy`), the global theme state,
  and the mapping from `light`/`dark` onto the `paper`/`monitor` attribute.
- `simulationContext.jsx` owns synthetic scenarios, patient updates, and
  pause/reset controls. It is entirely client-side.
- `api.js` centralizes the backend base URL and small `fetch` helpers.
- `theme/tokens.css` is the single source of colour, type, space, radius, and
  motion for both display modes.
- `motion/gsap.js` registers the GSAP plugins and exports the `REDUCED` /
  `MOTION_OK` media queries that every animation must gate on.
- `styles/` holds the visual system, split by surface: base, layout, landing,
  film, station, pages.
- `components/` contains route-level views plus `trace.js`, which owns the drawn
  ECG path and the risk-tier thresholds.

See `THEME.md` for the full design system.

### Route Map

| Route | View | Current data source |
|---|---|---|
| `/` | Landing page | Static content, hero film, and local preview animation |
| `/dashboard` | Patient dashboard | Synthetic simulation; one offline metrics request |
| `/simulated-data` | Simulated patient feed | Synthetic simulation context |
| `/waveforms` | Sensor waveform view | Synthetic waveform transformations |
| `/training` | Training job list | Backend training-jobs endpoint |
| `/training/new` | Training configuration | Backend training-start endpoint on submit |
| `/training/:jobId` | Training monitor | Backend training-job endpoint with polling |
| `/architecture` | Architecture narrative | Static project documentation |

There is currently no explicit not-found route, so unknown URLs render the normal application shell without a route-specific error.

## Data and Trust Boundaries

### Synthetic Dashboard Data

Patient records shown on the dashboard, simulated-data page, and waveform page are generated locally. Their risk values, alerts, NEWS2 comparison, impact signals, and lead-time estimates are demonstration calculations rather than outputs from the trained model.

The UI labels this behavior in several places, including the dashboard simulation banner and synthetic preview labels.

### Backend Communication

The simulation context can post synthetic vital samples to `/ingest`. The dashboard's patient queue, however, is not populated from the backend's scored results. This means the browser can send demonstration data to the backend while continuing to display its own local risk simulation.

Training views use the backend directly for job creation, job status, and metrics.

### Offline Model Metrics

The model snapshot requests `/metrics` when the dashboard mounts. The landing and architecture pages also present verified offline evaluation information, including the current research result of 0.844 holdout AUC with its stated limitations.

The holdout was used during ensemble selection and is not a locked final test set. It should continue to be described as a research evaluation result, not clinical validation or prospective evidence.

### Security and Deployment Status

The current application is a local research prototype. It does not provide authentication, authorization, encryption, audit logging, clinical validation, or a formal compliance boundary. Those limitations must remain visible if the interface is presented outside a controlled demonstration.

## What the Journey Established Well

- A clear route from project research to interactive product demonstration.
- A reusable queue-first dashboard structure.
- A consistent distinction between simulation controls and clinical-looking metrics.
- A recognizable visual identity with shared light and dark themes.
- Accessible focus, skip navigation, landmarks, labels, and reduced-motion behavior.
- Route-level lazy loading and local font assets.
- Honest research-prototype language that became stronger over time.
- A clear need to preserve backend/model/frontend feature parity.

## Engineering Lessons

### 1. Product Truth Must Be Structural

Warnings are useful, but they should not be the only protection against misrepresentation. Data source, timestamp, model version, and generation method should be represented in the application state and rendered consistently.

### 2. A Unified API Client Should Replace Mixed Request Styles

The dashboard and simulation context use `fetch`, while training views use Axios. This creates different timeout, cancellation, validation, and error-handling behavior. A shared client should normalize those concerns.

### 3. Shared Contracts Need Automated Verification

The training interface currently initializes twelve deployed features but displays only seven options, including an option that is not part of the documented deployed feature set. A runtime schema, generated types, or contract test should prevent frontend and backend configuration from drifting.

### 4. Accessibility Must Continue Past Manual Review

Semantic controls and reduced-motion behavior provide a strong baseline. Automated checks, keyboard walkthroughs, screen-reader review, zoom testing, and contrast regression tests are still required.

### 5. Demonstrations and Production Should Share Concepts, Not Labels

Simulation and live modes can share visualization components, but they should use separate data contracts, headings, timestamps, and status language. A single label such as "live" must never be used for browser-generated data.

## Current Risks and Technical Gaps

| Priority | Area | Finding | Required response |
|---|---|---|---|
| P0 | Data integrity | The dashboard displays local simulated risk rather than the score returned by backend inference | Separate simulation and live data modes at the state and component levels |
| P0 | Feature contract (serving) | The 12-feature order is duplicated in `ml/train.py` and `backend/inference.py` and is only cross-checked under `--deploy`, so a plain run can produce an unservable model | Import one shared constant; add a test that asserts the scaler feature order matches |
| P0 | Reliability | No route error boundary, explicit 404 view, or consistent request timeout/cancellation | Add failure containment and predictable recovery paths |
| P1 | API consistency | Request behavior differs between `fetch` and Axios | Consolidate transport, timeout, JSON, and error handling in one client |
| P1 | Testing | No frontend unit, integration, accessibility, or end-to-end suite | Add focused tests for simulation math, API states, routes, and critical workflows |
| P1 | Type safety | JSX and JavaScript allow invalid data shapes to reach render code | Introduce TypeScript incrementally, starting with API and simulation models |
| P1 | Dependency security | `npm audit` reports four advisories: one high and three moderate, affecting Vite, esbuild, and React Router | Plan and test major-version upgrades; do not apply `npm audit fix --force` without compatibility review |
| P1 | Theme consistency | *(resolved)* the Tailwind override layer was replaced by `theme/tokens.css` plus `src/styles/*.css` in the ICU-instruments redesign | Done — see `THEME.md` |
| P2 | Motion consistency | *(resolved)* broad `transition: all` utilities were removed; animations now gate on `MOTION_OK` from `motion/gsap.js` | Done |
| P2 | Routing and deployment | BrowserRouter requires correct SPA fallback configuration | Document and test production hosting rewrites |
| P2 | Observability | Frontend request failures are mostly shown locally or logged to the console | Add structured error reporting for deployed research environments |

## Prioritized Next Roadmap

### Phase 1: Protect Meaning and Reliability

1. Introduce a typed data-source model such as `simulation`, `backend-live`, and `offline-evaluation`.
2. Display source, generation time, and model version consistently in relevant views.
3. Stop the simulation from presenting locally generated risk as backend inference.
4. Replace the feature list in `TrainingConfig` with a backend-provided or shared canonical list.
5. Add a 404 route and route-level error boundary.
6. Add request timeouts, cancellation for obsolete polling, and safe responses for non-JSON errors.

### Phase 2: Establish a Test Safety Net

1. Add unit tests for simulation updates, scenario changes, risk status, and NEWS2 calculations.
2. Add component tests for loading, empty, success, and backend-error states.
3. Add route-level integration tests for dashboard, training setup, job monitoring, and navigation.
4. Add accessibility checks for landmarks, labels, keyboard behavior, focus visibility, and reduced motion.
5. Add a small end-to-end smoke suite for the highest-value user journeys.

### Phase 3: Strengthen the Engineering System

1. Add ESLint and enforce it in local development and CI.
2. Introduce TypeScript without requiring an unsafe full-project rewrite.
3. Generate API types or validate API payloads at runtime.
4. Consolidate backend access through one client.
5. Standardize route loading and error UI.
6. Add a production build and preview check to CI.
7. Upgrade Vite and React Router through reviewed compatibility changes, then re-run the full frontend test and build suite.

### Phase 4: Prepare for Controlled Deployment

1. Add authentication and authorization appropriate to a non-clinical research environment.
2. Add transport security and deployment configuration.
3. Define audit logging and retention requirements before using any real patient-related data.
4. Complete responsive, browser, zoom, contrast, and assistive-technology testing.
5. Establish explicit release ownership for model manifests, feature schemas, and frontend compatibility.

## Definition of Done for the Next Major Frontend Release

The next release should not be considered complete until:

- simulation, live inference, and offline metrics are visibly and structurally distinct;
- frontend and backend feature configurations cannot silently drift;
- unknown routes, network failures, invalid responses, and component crashes have predictable UI;
- critical simulation and training logic has automated tests;
- the application passes lint, type checking, production build, and end-to-end smoke checks;
- high or critical dependency advisories are resolved or have a reviewed, time-bounded exception;
- light and dark themes use the same semantic token system;
- keyboard-only and reduced-motion journeys are verified;
- deployment instructions include SPA route fallback behavior;
- all clinical, performance, security, and deployment claims match verified project evidence.

## Local Development Reference

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite development URL shown in the terminal. To verify a production build:

```powershell
cd frontend
npm run build
npm run preview
```

The current `package.json` provides development, build, and preview scripts, but it does not yet provide dedicated lint, type-check, or test commands.

## Closing Reflection

The strongest part of the SynCura frontend journey is not any single screen; it is the gradual refinement of the product's boundaries. The application began as a way to prove that the ML pipeline could be interacted with, expanded into a branded monitoring prototype, and then matured into a design system with accessibility, motion, and research-integrity safeguards.

The next stage should apply the same discipline to data and reliability. Once simulation, backend inference, and offline evaluation are structurally distinct, and critical workflows are protected by tests, the frontend can evolve from a compelling demonstration into a trustworthy research interface without losing the clarity that made it effective.
