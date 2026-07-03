# Website Restyle Changelog — "Obsidian Minimal" (July 2026)

Branch: `website-restyle`. Every commit on this branch is logged here: what changed, why, files touched, verification.
Direction: neutral true-dark competition-platform aesthetic (#0E0F11 bg, single #E5502B ember accent, hairline borders, flat surfaces, system fonts, tight type scale). Hard constraints: **no URL ever dies** (press links) and **seamless mobile** (360–768px verified).

Pipeline: multi-agent (design audit from full-page screenshots → token foundation → parallel work packages → QA gates: build, route crawl, URL preservation, console, page weight, bundle size, Tailwind purge tripwire, mobile compatibility, visual sign-off).

---

## C0 — `daeadb1b7` fix broken empty poster ref (404)

**Why:** the only console error on the baseline site: `docs/26_grandprize.md` referenced `poster="/img/grandprize/.webp"` (empty basename) — a 404 on every `/grandprize` load, 20 errors across QA viewports. No poster file ever existed; the video autoplays.
**Files:** `docs/26_grandprize.md` (1 line).
**Verification:** baseline capture identified it; the fix removes the attribute; post-fix expectation is zero console errors site-wide.

## C1 — tokens: design-token foundation, Tailwind bridge, content-glob fix

**Why:** establish one source of truth for the new visual system before any component work; fix a latent Tailwind purge bug.
**What:**
- New `src/css/tokens.css` — all `--vc-*` design tokens (Obsidian palette, type, radii, spacing). New `src/css/utilities.css` — shared recipes (`.vc-card`, `.vc-cta`, `.vc-btn`, `.vc-stat-strip`, `.vc-admonition`, `.vc-kicker`, `.vc-media`, …). New `chrome.css`/`landing.css` stubs for the parallel work packages; CSS load order: tokens → utilities → custom → chrome → landing → imageZoom.
- `tailwind.config.js`: content globs now cover `.mdx`/`.jsx` (previously `.mdx`-only classes survived purge by coincidence — a hard prerequisite for the upcoming stale-doc removal); color bridge `bg/surface/raised/line/dim/faint/accent/gold` → `var(--vc-*)`.
- `src/css/custom.css`: every raw color re-pointed to tokens (this re-skins the whole site to #0E0F11/#E5502B); global heading reset 900/-0.05em → 700/-0.01em (h1–h2) and 650/0 (h3+); docs heading sizes per new type scale; sidebar width 230→260px; Atlas block untouched except `--card`/`--line` indirection (its gold/status colors are frozen data-vis).
**Files:** `tailwind.config.js`, `docusaurus.config.js` (customCss array only), `src/css/{tokens,utilities,chrome,landing,custom}.css`, `RESTYLE_CHANGELOG.md`.
**Verification:** `yarn build` exit 0; old `#1C1A1D` gone from src CSS; `.mdx` purge tripwire class present in emitted bundle; `--vc-*` tokens compiled.
**Notable mappings:** Infima 7-step red ramp collapsed to the single ember accent; code-line highlight now a raised surface instead of a dark tint; `firstletters` black bg → `--vc-bg`.

## Baseline (pre-restyle, recorded 2026-07-03)

- `yarn build` green; 82 sitemap routes all HTTP 200.
- Landing (desktop, full scroll-flush): **15,248,955 bytes / 137 requests**.
- `build/` 624,524 KB; `build/assets` 22,824 KB.
- Console errors: 20 (all the C0 404). Mobile: no horizontal overflow; ~10 sub-44px tap targets per page (hamburger 30×30, footer links 29–35×32).
- 72 arbitrary-value Tailwind classes inventoried across docs (purge tripwire baseline); 1 pre-existing purge victim (`w-[70%]`, mdx-only) recorded.
