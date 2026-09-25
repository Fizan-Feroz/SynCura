# THEME.md — SynCura frontend design system

Current-state reference for the ICU-instruments redesign (commits `4c38499` and
`546e89c`, "redesign as ICU instruments (chart paper / bedside monitor)").

The organising idea is that the UI is a **clinical instrument**, not a dashboard
template. Two display modes mirror the two ways clinicians actually read a
monitor, and the whole token set is built on that split.

---

## 1. The two modes

`App.jsx` holds React state (`'light'` / `'dark'`) and maps it to a
`data-theme` attribute on `<html>`:

| React state | `data-theme` | Reads as | Default |
|---|---|---|---|
| `light` | `paper` | ECG chart paper: red millimetre grid, teal-black ink | yes |
| `dark` | `monitor` | Bedside monitor: dark screen, standard parameter colours | no |

Both token blocks are declared in `src/theme/tokens.css`. The `paper` values are
defined twice, on `:root` and on `[data-theme='paper']` (lines 8 and 45), so
they apply before React hydrates; `monitor` is the `[data-theme='monitor']` block
at line 78. `color-scheme` is set per mode, so native form controls and
scrollbars follow.

**Resolution order** (`App.jsx:15-23`, `initialTheme`):
1. `localStorage['syncura-theme']` if it is exactly `'dark'` or `'light'`
2. otherwise `prefers-color-scheme: dark` → `'dark'`, else `'light'`
3. both wrapped in `try/catch` because storage can throw in private mode

`App.jsx:45-52` writes the `data-theme` attribute and persists on every change.
The switch itself is `ThemeSwitch` in `AppShell.jsx`, a `role="group"`
segmented control with `aria-pressed` on each button and the labels **Paper** and
**Monitor**, not "light"/"dark".

---

## 2. Colour

### Surfaces and ink (paper mode)
`--paper #fafbfb` · `--surface #ffffff` · `--surface-2 #f2f5f5` ·
`--ink #0f2a33` · `--ink-soft #4b6470` · `--ink-faint #7d9099` ·
`--line #dce5e7` · `--line-strong #b9c9cd` · `--brand #0b6e80` (brand-ink `#fff`)

The grid is the identity of paper mode: `--grid-minor rgba(200,70,70,.075)` and
`--grid-major rgba(200,70,70,.16)`, a red millimetre grid over a near-white
sheet, with `--trace #0f2a33` drawing the waveform in the same ink as the text.

### Surfaces and ink (monitor mode)
`--paper #03100f` · `--surface #071a1a` · `--surface-2 #0b2322` ·
`--ink #e3f4ef` · `--ink-soft #95b4ac` · `--ink-faint #6a8a83` ·
`--line #143331` · `--line-strong #23504b` · `--brand #43d6c4` (brand-ink `#03100f`)

Grid drops to `rgba(67,214,196,.045)` / `.1` teal, `--trace` flips to `#3df08e`
phosphor green.

### Risk tiers — identical names in both modes
`--stable` · `--watch` · `--high` · `--critical`, each with a paired tint
`--stable-bg` · `--watch-bg` · `--critical-bg`. Paper uses flat low-chroma fills
(`#e5f2ee`, `#f8efdc`, `#f9e3e6`); monitor uses translucent overlays
(`rgba(63,214,164,.12)` etc.) so the dark grid stays visible underneath.

Thresholds are **not** in CSS. They live in `components/trace.js`:

```js
risk >= 85 -> critical
risk >= 70 -> high
risk >= 45 -> watch
else      -> stable
```

Keep the logic in `riskTone` / `riskLabel`; do not re-derive tiers in a component.

### Vital-sign colours
Five per-mode tokens, so each parameter keeps its identity across modes:

| Parameter | paper | monitor |
|---|---|---|
| `--hr` | `#16864c` green | `#3df08e` |
| `--spo2` | `#0a7fa0` | `#3cd7ff` |
| `--rr` | `#9a7200` | `#ffd447` |
| `--temp` | `#7447c4` | `#c4a7ff` |
| `--bp` | `#c4223f` | `#ff4d6a` |

`--bp` is deliberately the same red as `--critical` in paper mode; that is
intentional overlap, not a bug.

### Focus
`--focus` mirrors `--brand` in both modes (`#0b6e80` / `#43d6c4`).

---

## 3. Type

**One family: Archivo Variable**, self-hosted via
`@fontsource-variable/archivo` (`wdth.css` import in `main.jsx:4`). No webfont
request, no FOUT dependency on a CDN.

The width axis does the work instead of a second family:

| Token | `font-stretch` | Used for |
|---|---|---|
| `--wide` | 118% | headlines, section titles |
| `--normal` | 100% | body (set on `body`, `base.css:24`) |
| `--narrow` | 66% | numeric readouts and instrument labels |

Intermediate values (100–118%) appear in 8 places for fine tuning rather than
snapping to the three tokens. Monospace is used in exactly one place,
`layout.css:232` (`ui-monospace, 'Cascadia Code', Consolas`), for tabular
machine strings.

### Type scale
Roughly 1.333, with two fluid steps for large display text:

```
--t-xs  0.8125rem   13    labels, table headers
--t-sm  0.9375rem   15    secondary
--t-md  1.0625rem   17    body (set on body)
--t-lg  1.3125rem   21
--t-xl  1.75rem     28
--t-2xl 2.5rem      40    page titles
--t-3xl clamp(2.75rem, 6.2vw, 5.75rem)
--t-read clamp(2.25rem, 3.4vw, 3.25rem)   big vital readouts
```

Body is `--t-md` at `line-height: 1.55`.

---

## 4. Space, radius, motion, layout

**Space** is a 9-step scale: `--space-1` 4px through `--space-9` 112px
(4 / 8 / 12 / 16 / 24 / 32 / 48 / 72 / 112).

**Radius** is 3px (`--radius`) — instrument panels are nearly square. The only
exception is `--radius-pill` (999px), restricted to segmented controls.

**Motion:**
```css
--ease: cubic-bezier(0.22, 1, 0.36, 1);
--fast: 140ms;
--normal-dur: 260ms;
```
`body` transitions only `background-color` and `color` across the mode switch.
There is no `transition: all` anywhere.

**Layout:** `--content: 1240px`, `--gutter: clamp(16px, 4vw, 48px)`.

---

## 5. File map

Load order is fixed in `src/main.jsx:4-11` and must stay in this order —
`tokens.css` defines the custom properties every later sheet consumes.

| File | Role |
|---|---|
| `theme/tokens.css` | all design tokens; the only place with hex colour values |
| `styles/base.css` | reset, body defaults, focus, utility + notice classes |
| `styles/layout.css` | app shell, topbar, grid, skip link, monospace block |
| `styles/landing.css` | landing page |
| `styles/film.css` | hero film |
| `styles/station.css` | dashboard / central station |
| `styles/pages.css` | training, waveforms, data feed, architecture |
| `motion/gsap.js` | GSAP registration + `REDUCED` / `MOTION_OK` queries |
| `components/trace.js` | `ecgPath`, `seriesPath`, `riskTone`, `riskLabel` |

Tailwind is **not** in the build. `postcss.config.js` and `autoprefixer` are the
only PostCSS pieces; there is no `tailwind.config.js` and no Tailwind dependency
in `package.json`. Several root docs still describe a Tailwind + `tokens.css`
setup, which is stale — see "Known drift" below.

---

## 6. Rules that hold the system together

**Colour.** Semantic tokens only. `tokens.css` is the single source of hex.
The exception is the hero film (below).

**Type.** One family. Change the width axis for hierarchy, never swap in a
second typeface.

**Radius.** 3px everywhere; pill for segmented controls only.

**Motion.** `REDUCED` / `MOTION_OK` from `motion/gsap.js` gate every animation.
`LandingPage.jsx:62` and `ArchitecturePage.jsx:34` register inside
`mm.add(MOTION_OK, ...)` so animation is set up **only** when
`prefers-reduced-motion: no-preference` matches; `Dashboard.jsx:204` returns
early on `REDUCED`. `base.css:332` additionally collapses all transition and
animation durations to `0.01ms` under `reduce`. A new animation that skips the
media query is a bug.

**Accessibility.** Skip link to `#main-content` in `AppShell.jsx`; `main` has
`tabIndex="-1"`; `:focus-visible` is defined in four sheets; the theme switch
uses `aria-pressed`; live status uses `aria-live="polite"`; table headers use
`scope="col"`; loading state uses `role="status"` with `.sr-only` text.

**Honesty.** The simulation banner, research-prototype qualifier, and
not-HIPAA-ready note stay visible. Do not add metrics, certifications, or
clinical claims that the backend does not produce. The vitals shown are
client-side simulation (`simulationContext.jsx`), not a patient feed.

---

## 7. Two deliberate exceptions

**The hero film is a dark cinema stage in both modes.** `film.css` sets
`background: #020808` and defines local `--film-ink: #e8f6f1` /
`--film-accent: #7cf2e2`, plus 8 more hardcoded hex values for the SVG
scenery. It is a cinematic surface that cuts through medical materials, lands on
the SynCura horizon, then pulls back into a risk ring on chart paper on scroll.
These hex values are intentional and confined to that one file — do not
"fix" them into tokens.

**The ECG trace is drawn, not sampled.** `trace.js` `ecgPath` generates a Lead II
shape (P, Q, R, S, T) with a seeded LCG for jitter, so waveforms are stable
across renders. `seriesPath` is the real-data variant.

---

## 8. Known drift

These are documented as current state, not fixes:

- Root docs (`AGENTS.md`, `CLAUDE.md`) describe a Tailwind build with
  `frontend/tailwind.config.js`, `theme/tokens.css` + page-local colour systems,
  and `frontend/src/welcome.css`. None of those exist now. The real build is
  plain CSS + PostCSS/autoprefixer, the tokens are the only colour system, and
  the stylesheets are the six files in `styles/`.
- `App.jsx` still uses `light`/`dark` as React state while CSS uses
  `paper`/`monitor`. The mapping is in one place (`App.jsx:46`) — keep it there.
- `components/` contains a `.jsx` per route, all lazy-loaded via `React.lazy`
  in `App.jsx:6-13`.

---

## 9. Changing the theme

- **Colour, type, space, motion, radius** → `theme/tokens.css` only.
- **A new page or section** → add rules to the matching sheet in `styles/`
  (or a new one, imported after `tokens.css` in `main.jsx`).
- **A new animation** → register inside `MOTION_OK` from `motion/gsap.js`.
- **Never** introduce a second colour system, a second typeface, `transition: all`,
  or a hardcoded hex outside `tokens.css` and `film.css`.
