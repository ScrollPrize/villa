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

## C4 — nav + IA: real navbar, slim sidebar, TOC restoration, orphan pruning with redirects

**Why:** all navigation was crammed into one 35-link sidebar (external socials included) behind a hidden-navbar hack; ~⅓ of entries were closed-prize history advertising stale deadlines; long pages had no table of contents; three orphaned/duplicate docs added clutter.
**What:**
- **Real navbar** (was `items: []` + brand faked inside the sidebar): Prizes · Data · Tutorials · Milestones · Community ▾ (Discord/𝕏/Substack/Donate/Jobs) · right-aligned "Get Started" CTA (`vc-navbar-cta`). Brand-in-sidebar HTML hack deleted; desktop navbar-height-0 hack removed from custom.css (navbar height now 56px); leftover brand 50%-opacity dimming removed.
- **Sidebar** 15 → 8 top-level entries, zero external links; Milestones/Scrolls/Archive collapsed (`sidebarCollapsible` flipped to true); `ink_detection` + `grand_prize` moved into Archive (both closed 2023 prizes). Stale nav labels cleaned via `sidebar_label` (e.g. "$700k/$100k/$50k Grand Prize (Dec 31)" → "2023 Grand Prize (closed)").
- **TOC restored** on 16 long-form docs (`hide_table_of_contents` removed): master_plan, faq, all tutorials, winners, prizes, segmentation, livestream, grandprize, firstscroll, firstletters, community_projects. FAQ's hand-rolled `<TOCInline>` red-link index removed (the real TOC replaces it).
- **Orphans deleted WITH redirects** (no URL dies): `06_tutorial_thaumato.md` + `29_tutorial4.md` (duplicate "Segmentation - a different approach") → `/segmentation`; `open_problem_rep.md` → `/unwrapping`. 3 inbound links that pointed at `tutorial4` while talking about ink detection now point at the actual Ink Detection tutorial (`tutorial5`) — a pre-existing mislink, fixed.
- **Footer**: 3-column sitemap → one dense band (© + CC BY-NC · Discord/GitHub/Substack/𝕏/Jobs). GitHub link → `github.com/ScrollPrize`.
**Files:** `docusaurus.config.js`, `sidebars.js`, `src/css/custom.css` (navbar hack removal only), `docs/*` (frontmatter/labels, FAQ index, 2 link fixes, 3 deletions).
**Verification:** `yarn build` green under strict `onBrokenLinks: throw`; all 7 redirect routes generate; Discord URL verified against repo (`discord.gg/V4fJhvtaQn`).
**Known follow-ups:** `static/llms.txt` still references `/open_problem_rep` (URL stays alive via redirect; text updated in C5); grandprize page still carries its inline gradient h1 (P2 polish, iteration round).

## Baseline (pre-restyle, recorded 2026-07-03)

- `yarn build` green; 82 sitemap routes all HTTP 200.
- Landing (desktop, full scroll-flush): **15,248,955 bytes / 137 requests**.
- `build/` 624,524 KB; `build/assets` 22,824 KB.
- Console errors: 20 (all the C0 404). Mobile: no horizontal overflow; ~10 sub-44px tap targets per page (hamburger 30×30, footer links 29–35×32).
- 72 arbitrary-value Tailwind classes inventoried across docs (purge tripwire baseline); 1 pre-existing purge victim (`w-[70%]`, mdx-only) recorded.
